"""Training-only smooth reward for the unmodified ManiSkill peg task."""

from __future__ import annotations

import torch
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env


@register_env("PegInsertionSideSmoothReward-v1", max_episode_steps=100)
class PegInsertionSideSmoothRewardEnv(PegInsertionSideEnv):
    """Keep native dynamics and success, but smooth the insertion reward gate."""

    def __init__(self, *args, insertion_shaping_coefficient: float = 1.0, **kwargs):
        self.insertion_shaping_coefficient = float(insertion_shaping_coefficient)
        if self.insertion_shaping_coefficient < 0:
            raise ValueError("insertion_shaping_coefficient must be nonnegative")
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs, action: torch.Tensor, info: dict):
        reward = super().compute_dense_reward(obs, action, info)
        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
        peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
        yz_error = (
            torch.linalg.vector_norm(peg_head_wrt_goal.p[:, 1:], dim=1)
            + torch.linalg.vector_norm(peg_wrt_goal.p[:, 1:], dim=1)
        )
        # The native reward exposes insertion progress only after two hard
        # 1 cm tests pass.  This smooth gate supplies the same geometric
        # distance signal as alignment approaches that boundary, eliminating
        # the pre-insertion local optimum without changing dynamics or success.
        alignment_gate = torch.exp(-50.0 * yz_error)
        peg_head_in_hole = self.box_hole_pose.inv() * self.peg_head_pose
        insertion_reward = 5.0 * (
            1.0
            - torch.tanh(
                5.0 * torch.linalg.vector_norm(peg_head_in_hole.p, dim=1)
            )
        )
        reward += (
            self.insertion_shaping_coefficient
            * insertion_reward
            * alignment_gate
            * is_grasped
        )
        # Retain the benchmark's exact success reward and predicate.
        reward[info["success"]] = 10.0
        return reward
