#!/usr/bin/env python3
"""Compose a frozen one-goal policy twice to solve the ordered two-goal task."""

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


def hierarchical_observation(obs):
    extra = dict(obs["extra"])
    progress = extra["goal_progress"].bool()
    original_first = extra["instruction"].argmax(dim=1)
    rows = torch.arange(progress.shape[0], device=progress.device)
    first_complete = progress[rows, original_first]
    active = torch.where(first_complete, 1 - original_first, original_first)
    extra["instruction"] = torch.nn.functional.one_hot(active, 2).float()
    # Each invocation is a fresh one-goal subproblem. Ordered acceptance is
    # still enforced by the environment's untouched task memory.
    extra["goal_progress"] = torch.zeros_like(extra["goal_progress"])
    return flatten_state_dict({"agent": obs["agent"], "extra": extra}, use_torch=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--checkpoint", default=(
        "results/learned_recovery_v4/learned_recovery_v4_temporary_curriculum_transfer/"
        "temporary_one_goal_v1_transfer_ppo/seed_9351/stage1_frozen_safe.pt"
    ))
    parser.add_argument("--output", default="results/v4_hierarchical_one_goal_temp.json")
    args = parser.parse_args()
    env = gym.make(
        "LearnedRecovery-v4", num_envs=args.num_envs, obs_mode="state_dict",
        render_mode=None, sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1, required_goals=2,
        intervention_probability=1.0, intervention_types=("temporary_block",),
        onset_step_range=(0, 0), blocker_force=4.0, blocker_return_force=5.0,
        blocker_return_delay_steps=30, terminate_on_violation=True,
        safety_proximity_weight=5.0, constraint_violation_penalty=20.0,
    )
    if isinstance(env.action_space, gym.spaces.Dict): env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=False)
    obs, _ = env.reset(seed=253_000_000); state = hierarchical_observation(obs)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    agent = Agent(state.shape[1], int(np.prod(env.single_action_space.shape))).cuda()
    agent.load_state_dict(checkpoint["agent"]); agent.eval()
    successes = violations = 0
    try:
        for offset in range(0, args.episodes, args.num_envs):
            obs, _ = env.reset(seed=253_000_000 + offset)
            success = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
            violation = torch.zeros_like(success)
            for _ in range(240):
                action = agent.get_action(
                    hierarchical_observation(obs), deterministic=True,
                ).clamp(-1, 1)
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool(); violation |= info["constraint_violated"].bool()
            successes += int(success.sum()); violations += int(violation.sum())
    finally: env.close()
    result = {
        "schema_version": 1, "method": "hierarchical_repeated_one_goal_policy",
        "condition": "temporary_block", "episodes": args.episodes,
        "successes": successes, "success_rate": successes / args.episodes,
        "violations": violations, "violation_rate": violations / args.episodes,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
