#!/usr/bin/env python3
"""Audit whether every V4 physical intervention reaches its intended state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401


KINDS = ("ejection", "permanent_block", "temporary_block", "reverse_ejection")


def evaluate_kind(kind: str, *, episodes: int, num_envs: int, seed_base: int) -> dict:
    env = gym.make(
        "LearnedRecovery-v4", num_envs=num_envs, obs_mode="rgb", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        intervention_probability=1.0, intervention_types=(kind,),
        onset_step_range=(0, 0), intervention_force=6.0,
        intervention_steps=24, blocker_force=4.0,
        blocker_return_force=5.0, blocker_return_delay_steps=30,
        asymmetric_critic_observation=True, vision_camera_size=64,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=False)
    counts = {
        "target_unavailable": 0, "blocker_engaged": 0,
        "temporary_cleared": 0, "incorrect_non_target_unavailable": 0,
    }
    try:
        for offset in range(0, episodes, num_envs):
            obs, _ = env.reset(seed=seed_base + offset)
            action = torch.zeros(
                (num_envs,) + env.single_action_space.shape,
                device=obs["agent"]["qpos"].device,
            )
            info = None
            for _ in range(60):
                obs, _, _, _, info = env.step(action)
            target = env.unwrapped._intervention_target
            unavailable = env.unwrapped._recognized_unavailable()
            rows = torch.arange(num_envs, device=target.device)
            counts["target_unavailable"] += int(unavailable[rows, target].sum())
            non_target = 1 - target
            counts["incorrect_non_target_unavailable"] += int(
                unavailable[rows, non_target].sum()
            )
            counts["blocker_engaged"] += int(info["goal_blocker_engaged"].sum())
            counts["temporary_cleared"] += int(info["temporary_block_cleared"].sum())
    finally:
        env.close()
    return {"episodes": episodes, **counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=170_000_000)
    parser.add_argument("--output", default="results/v4_intervention_reliability_v1.json")
    args = parser.parse_args()
    if args.episodes % args.num_envs:
        raise ValueError("episodes must be divisible by num-envs")
    payload = {
        "schema_version": 1,
        "protocol": "zero-action physical intervention reliability audit",
        "results": {
            kind: evaluate_kind(
                kind, episodes=args.episodes, num_envs=args.num_envs,
                seed_base=args.seed_base + index * 1_000_000,
            )
            for index, kind in enumerate(KINDS)
        },
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
