"""Mechanism-diverse benchmark for irreversible task recovery.

V3 contains one failure mechanism: a force-driven sweeper ejects a requested
cube.  That is enough to test recovery control, but not enough to establish
that a policy learned *goal feasibility* rather than the appearance of one
specific event.  V4 keeps the same Panda manipulation task and adds two
force-driven goal-obstruction interventions:

``permanent_block``
    A heavy block is driven along the table into the selected goal pad and
    held there by a bounded force servo.  The cube can no longer occupy the accepted
    goal volume, so the selected goal becomes unavailable.

``temporary_block``
    The same block enters the goal pad and later retracts.  This is a hard
    negative for recovery: visual change alone must not authorize skipping.

No intervention assigns an actor pose.  Poses are set only during reset;
runtime changes are produced by forces and contact dynamics.  The environment
records when a block physically reaches a goal and makes irreversibility
monotone for the permanent intervention.  This is evaluator state and is only
exposed through explicitly privileged critic fields.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from atr.envs.learned_recovery_v3 import LearnedRecoveryEventRewardEnv


INTERVENTION_TYPES = (
    "ejection", "permanent_block", "temporary_block", "reverse_ejection",
)
EJECTION = 0
PERMANENT_BLOCK = 1
TEMPORARY_BLOCK = 2
REVERSE_EJECTION = 3


@register_env("LearnedRecovery-v4", max_episode_steps=240)
class LearnedRecoveryMechanismDiverseEnv(LearnedRecoveryEventRewardEnv):
    """V3 task with irreversible and reversible physical failure mechanisms."""

    blocker_half_sizes = (0.043, 0.043, 0.032)

    def __init__(
        self,
        *args,
        intervention_types: Sequence[str] = INTERVENTION_TYPES,
        blocker_force: float = 1.2,
        blocker_return_force: float = 1.6,
        blocker_return_delay_steps: int = 18,
        blocker_position_gain: float = 30.0,
        blocker_velocity_gain: float = 2.0,
        start_blocked: bool = False,
        start_blocked_x_range: Sequence[float] = (0.156, 0.156),
        start_blocked_velocity_range: Sequence[float] = (0.0, 0.0),
        control_delay_steps: int = 0,
        **kwargs,
    ):
        kinds = tuple(intervention_types)
        if not kinds:
            raise ValueError("intervention_types cannot be empty")
        unknown = set(kinds) - set(INTERVENTION_TYPES)
        if unknown:
            raise ValueError(f"unknown intervention types: {sorted(unknown)}")
        self.intervention_types = kinds
        self.blocker_force = float(blocker_force)
        self.blocker_return_force = float(blocker_return_force)
        self.blocker_return_delay_steps = int(blocker_return_delay_steps)
        self.blocker_position_gain = float(blocker_position_gain)
        self.blocker_velocity_gain = float(blocker_velocity_gain)
        self.start_blocked = bool(start_blocked)
        self.start_blocked_x_range = tuple(float(x) for x in start_blocked_x_range)
        self.start_blocked_velocity_range = tuple(
            float(x) for x in start_blocked_velocity_range
        )
        self.control_delay_steps = int(control_delay_steps)
        if len(self.start_blocked_x_range) != 2:
            raise ValueError("start_blocked_x_range must contain two values")
        if len(self.start_blocked_velocity_range) != 2:
            raise ValueError("start_blocked_velocity_range must contain two values")
        if self.control_delay_steps < 0:
            raise ValueError("control_delay_steps cannot be negative")
        if self.blocker_return_delay_steps < 1:
            raise ValueError("blocker_return_delay_steps must be positive")
        self._intervention_mechanism = None
        self._blocker_engaged = None
        self._temporary_cleared = None
        super().__init__(*args, **kwargs)

    def step(self, action):
        if self.control_delay_steps and isinstance(action, torch.Tensor):
            delayed = self._episode_step < self.control_delay_steps
            action = torch.where(delayed[:, None], torch.zeros_like(action), action)
        return super().step(action)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Each block begins just behind the camera plane and is driven to a
        # goal by a bounded force servo.  There are no extra static obstacles
        # in the nominal manipulation workspace.
        self.red_goal_blocker = actors.build_box(
            self.scene, self.blocker_half_sizes, [0.18, 0.18, 0.18, 1],
            "red_goal_blocker", initial_pose=sapien.Pose(p=[0.46, -0.25, 0.032]),
        )
        self.blue_goal_blocker = actors.build_box(
            self.scene, self.blocker_half_sizes, [0.18, 0.18, 0.18, 1],
            "blue_goal_blocker", initial_pose=sapien.Pose(p=[0.46, 0.25, 0.032]),
        )
        self.red_reverse_sweeper = actors.build_box(
            self.scene, [0.025, 0.042, 0.025], [0.25, 0.25, 0.25, 1],
            "red_reverse_sweeper", initial_pose=sapien.Pose(p=[0.46, -0.12, 0.025]),
        )
        self.blue_reverse_sweeper = actors.build_box(
            self.scene, [0.025, 0.042, 0.025], [0.25, 0.25, 0.25, 1],
            "blue_reverse_sweeper", initial_pose=sapien.Pose(p=[0.46, 0.12, 0.025]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # V3 initializes its reward potential through the virtual
        # _recognized_unavailable method, so these tensors must exist before
        # delegating to the parent reset.
        if self._intervention_mechanism is None:
            with torch.device(self.device):
                self._intervention_mechanism = torch.full(
                    (self.num_envs,), -1, dtype=torch.long
                )
                self._blocker_engaged = torch.zeros(
                    (self.num_envs, 2), dtype=torch.bool
                )
                self._temporary_cleared = torch.zeros(
                    self.num_envs, dtype=torch.bool
                )
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            batch = len(env_idx)
            choices = torch.tensor(
                [INTERVENTION_TYPES.index(kind) for kind in self.intervention_types],
                dtype=torch.long,
            )
            selected = choices[torch.randint(0, len(choices), (batch,))]
            has_intervention = self._intervention_target[env_idx] >= 0
            self._intervention_mechanism[env_idx] = torch.where(
                has_intervention, selected, torch.full_like(selected, -1)
            )
            self._blocker_engaged[env_idx] = False
            self._temporary_cleared[env_idx] = False

            start = torch.zeros((batch, 3))
            start[:, 0] = 0.46
            start[:, 2] = self.blocker_half_sizes[2]
            red = start.clone()
            red[:, 1] = -0.25
            blue = start.clone()
            blue[:, 1] = 0.25
            block_mechanism = (
                (self._intervention_mechanism[env_idx] == PERMANENT_BLOCK)
                | (self._intervention_mechanism[env_idx] == TEMPORARY_BLOCK)
            )
            start_engaged = self.start_blocked & has_intervention & block_mechanism
            target_red = start_engaged & (self._intervention_target[env_idx] == 0)
            target_blue = start_engaged & (self._intervention_target[env_idx] == 1)
            engaged_x = torch.empty(batch).uniform_(*self.start_blocked_x_range)
            red[:, 0] = torch.where(target_red, engaged_x, red[:, 0])
            blue[:, 0] = torch.where(target_blue, engaged_x, blue[:, 0])
            self.red_goal_blocker.set_pose(Pose.create_from_pq(red))
            self.blue_goal_blocker.set_pose(Pose.create_from_pq(blue))
            zero = torch.zeros((batch, 3))
            engaged_velocity = torch.empty(batch).uniform_(
                *self.start_blocked_velocity_range
            )
            red_velocity = zero.clone()
            blue_velocity = zero.clone()
            red_velocity[:, 0] = torch.where(target_red, engaged_velocity, red_velocity[:, 0])
            blue_velocity[:, 0] = torch.where(target_blue, engaged_velocity, blue_velocity[:, 0])
            self.red_goal_blocker.set_linear_velocity(red_velocity)
            self.blue_goal_blocker.set_linear_velocity(blue_velocity)
            self._blocker_engaged[env_idx, 0] = target_red
            self._blocker_engaged[env_idx, 1] = target_blue
            reverse = torch.zeros((batch, 3))
            reverse[:, 0] = 0.46
            reverse[:, 2] = 0.025
            red_reverse = reverse.clone()
            red_reverse[:, 1] = -0.12
            blue_reverse = reverse.clone()
            blue_reverse[:, 1] = 0.12
            self.red_reverse_sweeper.set_pose(Pose.create_from_pq(red_reverse))
            self.blue_reverse_sweeper.set_pose(Pose.create_from_pq(blue_reverse))
            self.red_reverse_sweeper.set_linear_velocity(zero)
            self.blue_reverse_sweeper.set_linear_velocity(zero)

    def _apply_batched_force(self, actor, force: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            actor.apply_force(force)
        else:
            actor.apply_force(force[0].detach().cpu().numpy().astype(np.float32))

    def _before_simulation_step(self):
        step = self._episode_step
        started = step >= self._onset_step
        ejection_active = (
            started
            & (step < self._onset_step + self.intervention_steps)
            & (self._intervention_mechanism == EJECTION)
        )
        reverse_ejection_active = (
            started
            & (step < self._onset_step + self.intervention_steps)
            & (self._intervention_mechanism == REVERSE_EJECTION)
        )
        target_red = self._intervention_target == 0
        target_blue = self._intervention_target == 1

        sweep = torch.zeros((self.num_envs, 3), device=self.device)
        sweep[:, 0] = self.intervention_force
        self._apply_batched_force(
            self.red_sweeper, sweep * (ejection_active & target_red)[:, None]
        )
        self._apply_batched_force(
            self.blue_sweeper, sweep * (ejection_active & target_blue)[:, None]
        )
        def reverse_servo(actor):
            raw = (
                15.0 * (-0.45 - actor.pose.p[:, 0])
                - 2.0 * actor.linear_velocity[:, 0]
            ).clamp(-self.intervention_force, self.intervention_force)
            force = torch.zeros((self.num_envs, 3), device=self.device)
            force[:, 0] = raw * reverse_ejection_active.float()
            return force
        self._apply_batched_force(
            self.red_reverse_sweeper,
            reverse_servo(self.red_reverse_sweeper) * target_red[:, None],
        )
        self._apply_batched_force(
            self.blue_reverse_sweeper,
            reverse_servo(self.blue_reverse_sweeper) * target_blue[:, None],
        )

        permanent_hold = started & (
            self._intervention_mechanism == PERMANENT_BLOCK
        )
        temporary_motion = (
            started
            & (self._intervention_mechanism == TEMPORARY_BLOCK)
        )
        blocking = permanent_hold | temporary_motion
        temporary_return = (
            (self._intervention_mechanism == TEMPORARY_BLOCK)
            & (step >= self._onset_step + self.blocker_return_delay_steps)
        )
        desired_x = torch.where(
            temporary_return,
            torch.full_like(step, 0.46, dtype=torch.float32),
            torch.full_like(step, 0.156, dtype=torch.float32),
        )
        def servo_force(actor):
            raw = (
                self.blocker_position_gain * (desired_x - actor.pose.p[:, 0])
                - self.blocker_velocity_gain * actor.linear_velocity[:, 0]
            )
            raw = raw.clamp(-self.blocker_force, self.blocker_return_force)
            force = torch.zeros((self.num_envs, 3), device=self.device)
            force[:, 0] = raw * blocking.float()
            return force
        self._apply_batched_force(
            self.red_goal_blocker,
            servo_force(self.red_goal_blocker) * target_red[:, None],
        )
        self._apply_batched_force(
            self.blue_goal_blocker,
            servo_force(self.blue_goal_blocker) * target_blue[:, None],
        )

    def _after_control_step(self):
        super()._after_control_step()
        blockers = torch.stack(
            [self.red_goal_blocker.pose.p, self.blue_goal_blocker.pose.p], dim=1
        )
        goals = torch.stack([self.red_goal.pose.p, self.blue_goal.pose.p], dim=1)
        near_goal = torch.linalg.norm(
            blockers[:, :, :2] - goals[:, :, :2], dim=2
        ) < 0.055
        block_mechanism = (
            (self._intervention_mechanism == PERMANENT_BLOCK)
            | (self._intervention_mechanism == TEMPORARY_BLOCK)
        )
        target = torch.nn.functional.one_hot(
            self._intervention_target.clamp_min(0), 2
        ).bool()
        valid = (self._intervention_target >= 0) & block_mechanism
        self._blocker_engaged |= near_goal & target & valid[:, None]

        target_x = blockers[
            torch.arange(self.num_envs, device=self.device),
            self._intervention_target.clamp_min(0), 0,
        ]
        self._temporary_cleared |= (
            (self._intervention_mechanism == TEMPORARY_BLOCK)
            & self._blocker_engaged.any(dim=1)
            & (target_x > 0.42)
        )

    def _recognized_unavailable(self) -> torch.Tensor:
        target = torch.nn.functional.one_hot(
            self._intervention_target.clamp_min(0), 2
        ).bool()
        valid = self._intervention_target >= 0
        ejected = (
            super()._recognized_unavailable()
            & ((self._intervention_mechanism == EJECTION)
               | (self._intervention_mechanism == REVERSE_EJECTION))[:, None]
        )
        permanently_blocked = (
            self._blocker_engaged
            & target
            & valid[:, None]
            & (self._intervention_mechanism == PERMANENT_BLOCK)[:, None]
        )
        return ejected | permanently_blocked

    def _unavailable(self) -> torch.Tensor:
        positions = torch.stack([self.red_cube.pose.p, self.blue_cube.pose.p], dim=1)
        return (positions[:, :, 0].abs() > 0.36) | (positions[:, :, 2] < -0.02)

    def evaluate(self):
        info = super().evaluate()
        info.update({
            "intervention_mechanism": self._intervention_mechanism,
            "goal_blocker_engaged": self._blocker_engaged.any(dim=1),
            "temporary_block_cleared": self._temporary_cleared,
            "permanent_goal_block": (
                self._intervention_mechanism == PERMANENT_BLOCK
            ),
            "temporary_goal_block": (
                self._intervention_mechanism == TEMPORARY_BLOCK
            ),
        })
        return info

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.asymmetric_critic_observation:
            obs.update({
                "critic_red_goal_blocker_pose": self.red_goal_blocker.pose.raw_pose,
                "critic_blue_goal_blocker_pose": self.blue_goal_blocker.pose.raw_pose,
                "critic_red_reverse_sweeper_pose": self.red_reverse_sweeper.pose.raw_pose,
                "critic_blue_reverse_sweeper_pose": self.blue_reverse_sweeper.pose.raw_pose,
                "critic_intervention_mechanism": torch.nn.functional.one_hot(
                    self._intervention_mechanism.clamp_min(0), len(INTERVENTION_TYPES)
                ).float(),
            })
        if "state" in self.obs_mode:
            # State-policy teachers see physical goal-blocker poses, but not
            # the intervention mechanism ID. Reverse ejection must therefore
            # transfer through the resulting cube state rather than a label.
            obs.update({
                "red_goal_blocker_pose": self.red_goal_blocker.pose.raw_pose,
                "blue_goal_blocker_pose": self.blue_goal_blocker.pose.raw_pose,
            })
        return obs
