#!/usr/bin/env python3
"""V4 DAgger/PPO with unavailable goals marked resolved for the state teacher."""

from mani_skill.utils.common import flatten_state_dict

import train_visual_recovery_dual_teacher_ppo as base


def reconstruct_resolved_teacher_observation(obs):
    """Present causally unavailable goals as resolved to the strict teacher."""
    extra = obs["extra"]
    state = {
        "agent": obs["agent"],
        "extra": {
            "tcp_pose": extra["tcp_pose"],
            "instruction": extra["instruction"],
            "goal_progress": base.visual_progress_target(obs),
            "red_cube_pose": extra["critic_red_cube_pose"],
            "blue_cube_pose": extra["critic_blue_cube_pose"],
            "red_goal_pos": extra["critic_red_goal_pos"],
            "blue_goal_pos": extra["critic_blue_goal_pos"],
            "red_sweeper_pose": extra["critic_red_sweeper_pose"],
            "blue_sweeper_pose": extra["critic_blue_sweeper_pose"],
            "protected_pose": extra["critic_protected_pose"],
        },
    }
    return flatten_state_dict(state, use_torch=True)


if __name__ == "__main__":
    base.reconstruct_state_teacher_observation = reconstruct_resolved_teacher_observation
    base.main()
