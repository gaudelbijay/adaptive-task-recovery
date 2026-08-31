#!/usr/bin/env python3
"""Verify RGB critic fields exactly reconstruct the V4 state-teacher vector."""

import gymnasium as gym
import torch

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
from train_v4_permanent_visual_dagger import (
    reconstruct_v4_state_teacher_observation,
)


def make(obs_mode: str):
    return gym.make(
        "LearnedRecovery-v4", num_envs=8, obs_mode=obs_mode,
        render_mode=None, sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
        reconfiguration_freq=1,
        asymmetric_critic_observation=(obs_mode == "rgb"),
        intervention_probability=1.0,
        intervention_types=("permanent_block",), onset_step_range=(0, 0),
        blocker_force=4.0, blocker_return_force=5.0,
        blocker_return_delay_steps=30, terminate_on_violation=True,
    )


def main():
    state_env, rgb_env = make("state"), make("rgb")
    try:
        state, _ = state_env.reset(seed=24681357)
        rgb, _ = rgb_env.reset(seed=24681357)
        reconstructed = reconstruct_v4_state_teacher_observation(rgb)
        if state.shape != reconstructed.shape:
            raise RuntimeError(
                f"shape mismatch: state={state.shape}, reconstructed={reconstructed.shape}"
            )
        maximum_error = float((state - reconstructed).abs().max())
        if not torch.allclose(state, reconstructed, atol=1e-6, rtol=1e-6):
            raise RuntimeError(f"teacher reconstruction mismatch: max_error={maximum_error}")
        print({"shape": list(state.shape), "maximum_absolute_error": maximum_error})
    finally:
        state_env.close()
        rgb_env.close()


if __name__ == "__main__":
    main()
