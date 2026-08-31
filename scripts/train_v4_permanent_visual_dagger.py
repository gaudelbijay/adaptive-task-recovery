#!/usr/bin/env python3
"""Distill the V4 permanent-block state specialist into a visual policy."""

from mani_skill.utils.common import flatten_state_dict

import train_visual_recovery_dual_teacher_ppo as base


def reconstruct_v4_state_teacher_observation(obs):
    """Rebuild LearnedRecovery-v4 state observations without mechanism labels."""
    extra = obs["extra"]
    state = {
        "agent": obs["agent"],
        "extra": {
            "tcp_pose": extra["tcp_pose"],
            "instruction": extra["instruction"],
            "goal_progress": extra["goal_progress"],
            "red_cube_pose": extra["critic_red_cube_pose"],
            "blue_cube_pose": extra["critic_blue_cube_pose"],
            "red_goal_pos": extra["critic_red_goal_pos"],
            "blue_goal_pos": extra["critic_blue_goal_pos"],
            "red_sweeper_pose": extra["critic_red_sweeper_pose"],
            "blue_sweeper_pose": extra["critic_blue_sweeper_pose"],
            "protected_pose": extra["critic_protected_pose"],
            "red_goal_blocker_pose": extra["critic_red_goal_blocker_pose"],
            "blue_goal_blocker_pose": extra["critic_blue_goal_blocker_pose"],
        },
    }
    return flatten_state_dict(state, use_torch=True)


if __name__ == "__main__":
    base.reconstruct_state_teacher_observation = (
        reconstruct_v4_state_teacher_observation
    )
    base.main()
