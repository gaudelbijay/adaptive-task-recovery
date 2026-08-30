"""Event-shaped revision of the non-teleport recovery benchmark.

``LearnedRecovery-v2`` paid the number of completed goals on every subsequent
step.  Under the training discount, that made delaying the terminal second
placement more valuable than completing the task.  V3 leaves the scene,
physics, observations, interventions, safety constraint, and success semantics
unchanged, but replaces that persistent term with bounded progress deltas and
one-time completion bonuses.

The class lives in a separate module so queued and archived V2 experiments keep
their exact registered implementation.  V2 remains useful as negative evidence
from the reward audit, but is not eligible for final performance claims.
"""

from __future__ import annotations

from typing import Any

import torch
from mani_skill.utils.registration import register_env

from atr.envs.learned_recovery import LearnedRecoveryEnv


def event_progress_reward(
    progress_delta: torch.Tensor,
    newly_completed: torch.Tensor,
    success: torch.Tensor,
    proximity_risk: torch.Tensor,
    constraint_violated: torch.Tensor,
    *,
    progress_scale: float,
    completion_bonus: float,
    success_reward: float,
    safety_proximity_weight: float,
    constraint_violation_penalty: float,
) -> torch.Tensor:
    """Compute V3 reward from transition-local quantities.

    Keeping this arithmetic separate makes the no-stalling invariant directly
    testable: unchanged progress with no new event has zero task reward.
    """

    reward = progress_scale * progress_delta
    reward = reward + completion_bonus * newly_completed
    reward = reward - safety_proximity_weight * proximity_risk
    reward = reward - constraint_violation_penalty * constraint_violated.float()
    return torch.where(success, torch.full_like(reward, success_reward), reward)


@register_env("LearnedRecovery-v3", max_episode_steps=200)
class LearnedRecoveryEventRewardEnv(LearnedRecoveryEnv):
    """V2 task semantics with transition-local, non-stalling dense reward."""

    def __init__(
        self,
        *args,
        progress_reward_scale: float = 2.0,
        completion_bonus: float = 5.0,
        success_reward: float = 10.0,
        **kwargs,
    ):
        self.progress_reward_scale = float(progress_reward_scale)
        self.completion_bonus = float(completion_bonus)
        self.success_reward = float(success_reward)
        self._reward_potential = None
        self._newly_completed = None
        super().__init__(*args, **kwargs)

    def _task_potential(self) -> torch.Tensor:
        """Bounded stage potential for the currently required ordered goal."""

        unavailable = self._recognized_unavailable()
        rows = torch.arange(self.num_envs, device=self.device)
        first = self._instruction_first
        second = 1 - first
        first_resolved = self._completed[rows, first] | unavailable[rows, first]
        active = torch.where(first_resolved, second, first)

        cube_positions = torch.stack(
            [self.red_cube.pose.p, self.blue_cube.pose.p], dim=1
        )
        goal_positions = torch.stack(
            [self.red_goal.pose.p, self.blue_goal.pose.p], dim=1
        )
        cube = cube_positions[rows, active]
        goal = goal_positions[rows, active]
        reaching = 1 - torch.tanh(
            5 * torch.linalg.norm(cube - self.agent.tcp.pose.p, dim=1)
        )
        grasped = torch.where(
            active == 0,
            self.agent.is_grasping(self.red_cube),
            self.agent.is_grasping(self.blue_cube),
        )
        placing = 1 - torch.tanh(5 * torch.linalg.norm(cube - goal, dim=1))
        return torch.where(grasped, 1.0 + placing, reaching)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if self._reward_potential is None:
            self._reward_potential = torch.zeros(self.num_envs, device=self.device)
            self._newly_completed = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        self._reward_potential[env_idx] = self._task_potential()[env_idx]
        self._newly_completed[env_idx] = 0

    def _update_task_memory(self):
        before = self._completed.clone()
        unavailable = super()._update_task_memory()
        self._newly_completed = (self._completed & ~before).sum(dim=1)
        return unavailable

    def compute_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        current_potential = self._task_potential()
        progress_delta = current_potential - self._reward_potential

        # A completion switches the active object, so the two stage potentials
        # are not comparable on that transition.  The one-time event bonus is
        # the complete signal for it; subsequent progress starts from the new
        # object's current potential.
        completion_event = self._newly_completed.float()
        progress_delta = torch.where(
            completion_event > 0, torch.zeros_like(progress_delta), progress_delta
        )

        tcp_clearance = torch.linalg.norm(
            self.agent.tcp.pose.p - self.protected.pose.p, dim=1
        )
        proximity_risk = torch.clamp((0.12 - tcp_clearance) / 0.12, min=0.0)
        reward = event_progress_reward(
            progress_delta,
            completion_event,
            info["success"],
            proximity_risk,
            self._constraint_violated,
            progress_scale=self.progress_reward_scale,
            completion_bonus=self.completion_bonus,
            success_reward=self.success_reward,
            safety_proximity_weight=self.safety_proximity_weight,
            constraint_violation_penalty=self.constraint_violation_penalty,
        )
        self._reward_potential = current_potential.detach()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / self.success_reward
