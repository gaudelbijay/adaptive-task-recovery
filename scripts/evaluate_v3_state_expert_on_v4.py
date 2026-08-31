#!/usr/bin/env python3
"""Evaluate the frozen V3 object-state recovery expert on V4 ejection physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.common import flatten_state_dict
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
from train_manipulation_ppo import Agent


CONDITIONS = ("nominal", "ejection", "reverse_ejection")


def v3_observation(obs):
    extra = {
        key: value for key, value in obs["extra"].items()
        if key not in ("red_goal_blocker_pose", "blue_goal_blocker_pose")
    }
    # The recovery decision is invariant to which side of the table an
    # already-unreachable cube exited. Canonicalize only physically
    # out-of-workspace cube coordinates; no mechanism label is consulted.
    for key in ("red_cube_pose", "blue_cube_pose"):
        pose = extra[key].clone()
        pose[:, 0] = torch.where(
            pose[:, 0].abs() > 0.36, pose[:, 0].abs(), pose[:, 0]
        )
        extra[key] = pose
    return flatten_state_dict({"agent": obs["agent"], "extra": extra}, use_torch=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint", default=(
        "results/learned_recovery/learned_recovery_ppo_v11_strict_removal/"
        "event_reward_strict_removal_state_ppo/seed_9351/best.pt"
    ))
    parser.add_argument("--output-dir", default="results/v3_state_expert_on_v4")
    args = parser.parse_args(); condition = CONDITIONS[args.task_index]
    probability = 0.0 if condition == "nominal" else 1.0
    env = gym.make(
        "LearnedRecovery-v4", num_envs=args.num_envs, obs_mode="state_dict",
        render_mode=None, sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        intervention_probability=probability,
        intervention_types=("ejection",) if condition == "nominal" else (condition,),
        onset_step_range=(0, 0), intervention_force=6.0, intervention_steps=24,
        blocker_force=4.0, blocker_return_force=5.0, blocker_return_delay_steps=30,
        terminate_on_violation=True, safety_proximity_weight=5.0,
        constraint_violation_penalty=20.0,
    )
    if isinstance(env.action_space, gym.spaces.Dict): env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=False)
    obs, _ = env.reset(seed=251_000_000)
    state = v3_observation(obs)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    agent = Agent(state.shape[1], int(np.prod(env.single_action_space.shape))).cuda()
    agent.load_state_dict(checkpoint["agent"]); agent.eval()
    successes = violations = 0
    try:
        for offset in range(0, args.episodes, args.num_envs):
            obs, _ = env.reset(seed=251_000_000 + offset)
            success = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
            violation = torch.zeros_like(success)
            for _ in range(240):
                action = agent.get_action(v3_observation(obs), deterministic=True).clamp(-1, 1)
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool(); violation |= info["constraint_violated"].bool()
            successes += int(success.sum()); violations += int(violation.sum())
    finally:
        env.close()
    result = {
        "schema_version": 2, "method": "symmetry_canonicalized_v3_state_expert",
        "condition": condition, "episodes": args.episodes,
        "successes": successes, "success_rate": successes / args.episodes,
        "violations": violations, "violation_rate": violations / args.episodes,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "heldout_mechanism": condition == "reverse_ejection",
    }
    output = Path(args.output_dir) / f"{condition}.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
