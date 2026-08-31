#!/usr/bin/env python3
"""Evaluate a frozen official-style PPO checkpoint on native PegInsertion.

This is the competence audit that precedes any external recovery/router work.
It uses deterministic actions, fresh development seeds, and no interventions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_manipulation_ppo import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=421_000_000)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes % args.num_envs:
        raise ValueError("episodes must be divisible by num-envs")
    if not torch.cuda.is_available():
        raise RuntimeError("PegInsertion PPO evaluation requires CUDA")

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    task = checkpoint["task"]
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])
    if float(task.get("env_kwargs", {}).get("intervention_probability", 1.0)) != 0.0:
        raise ValueError("competence checkpoint was not trained with interventions disabled")

    env_kwargs = {
        "obs_mode": task.get("obs_mode", "state"),
        "render_mode": None,
        "sim_backend": "physx_cuda",
        "control_mode": task["control_mode"],
        "reward_mode": task.get("reward_mode", "normalized_dense"),
        **task.get("eval_env_kwargs", task.get("env_kwargs", {})),
    }
    env_kwargs.update({
        "intervention_probability": 0.0,
        "intervention_types": ("positive_lateral_peg_ejection",),
    })
    env = gym.make(
        task["env_id"], num_envs=args.num_envs, reconfiguration_freq=1,
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, args.num_envs, ignore_terminations=True, record_metrics=False,
    )
    observation_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = Agent(observation_dim, action_dim).to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    agent.eval()
    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)

    successes = 0
    violations = 0
    try:
        for batch in range(args.episodes // args.num_envs):
            observation, _ = env.reset(seed=args.seed_base + batch)
            success_once = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            violation_once = torch.zeros_like(success_once)
            with torch.no_grad():
                for _ in range(args.steps):
                    action = torch.clamp(
                        agent.get_action(observation, deterministic=True),
                        action_low, action_high,
                    )
                    observation, _, _, _, info = env.step(action)
                    success_once |= info["success"].bool()
                    violation_once |= info["constraint_violated"].bool()
            successes += int((success_once & ~violation_once).sum().item())
            violations += int(violation_once.sum().item())
    finally:
        env.close()

    result = {
        "schema_version": 1,
        "audit": "official_ppo_nominal_competence",
        "environment": task["env_id"],
        "training_seed": int(task["seed"]),
        "evaluation_seed_base": args.seed_base,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "episodes": args.episodes,
        "safe_successes": successes,
        "safe_success_rate": successes / args.episodes,
        "violations": violations,
        "violation_rate": violations / args.episodes,
        "intervention_probability": 0.0,
        "runtime_pose_assignment": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
