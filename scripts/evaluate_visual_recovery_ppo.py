#!/usr/bin/env python3
"""Held-out deterministic evaluation for the restricted-input visual policy."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_visual_recovery_ppo import (
    VisualAgent, env_kwargs, extract_observation, metric_success, select_task,
)


def wilson(successes, trials, z=1.959963984540054):
    if trials == 0:
        return [float("nan"), float("nan")]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_ppo_gate_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=91000000)
    parser.add_argument("--condition", choices=("configured", "nominal", "intervention"), default="configured")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("visual policy evaluation requires CUDA")
    if args.episodes % args.num_envs:
        raise ValueError("episodes must be divisible by num-envs for exact paired evaluation")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = select_task(config, args.task_index)
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])
    seed = int(task["seed"])
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint unavailable: {checkpoint_path}")

    kwargs = env_kwargs(task, evaluation=True)
    if args.condition == "nominal":
        kwargs["intervention_probability"] = 0.0
    elif args.condition == "intervention":
        kwargs["intervention_probability"] = 1.0
    envs = gym.make(task["env_id"], num_envs=args.num_envs, reconfiguration_freq=1, **kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    observation, _ = envs.reset(seed=args.seed_base + seed * 100000)
    rgb, proprio, critic_state = extract_observation(observation, task["asymmetric_critic"])
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic_state.shape[1], action_dim,
        task["asymmetric_critic"], task.get("augmentation_pad", 0),
    ).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")
    if checkpoint.get("observation_contract") != "rgb_qpos_qvel_instruction_v1":
        raise ValueError("checkpoint lacks the restricted visual observation contract")
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    completed = 0
    episode_records = []
    tracked_maxima = (
        "goals_completed", "goals_unavailable", "constraint_violated",
        "intervention_occurred",
    )
    branches = ("first_goal_removed", "instruction_red_first")
    with torch.no_grad():
        while completed < args.episodes:
            # The completed offset and training seed are common across methods,
            # making episode records exactly pairable within each training seed.
            batch_seed = args.seed_base + seed * 100000 + completed
            observation, _ = envs.reset(seed=batch_seed)
            metrics = defaultdict(list)
            maxima = {key: torch.zeros(args.num_envs, device="cuda") for key in tracked_maxima}
            branch_values = {}
            for step in range(int(task["num_eval_steps"])):
                rgb, proprio, _ = extract_observation(observation, task["asymmetric_critic"])
                observation, _, _, _, info = envs.step(agent.get_action(rgb, proprio, True))
                if step == 0:
                    for key in branches:
                        if key in info:
                            branch_values[key] = info[key].detach().float().reshape(-1).clone()
                for key in tracked_maxima:
                    if key in info:
                        maxima[key] = torch.maximum(maxima[key], info[key].detach().float().reshape(-1))
                if "final_info" in info:
                    mask = info["_final_info"]
                    for key, value in info["final_info"]["episode"].items():
                        metrics[key].extend(value[mask].detach().float().cpu().tolist())
            available = max((len(values) for values in metrics.values()), default=0)
            take = min(available, args.episodes - completed)
            if take != args.num_envs:
                raise RuntimeError(f"expected {args.num_envs} completed episodes, observed {take}")
            for index in range(take):
                record = {key: float(values[index]) for key, values in metrics.items()}
                record.update({key: float(values[index]) for key, values in maxima.items()})
                record.update({key: float(values[index]) for key, values in branch_values.items()})
                episode_records.append(record)
            completed += take

    success_values = [metric_success(record) for record in episode_records]
    if any(math.isnan(value) for value in success_values):
        raise RuntimeError("success metric missing from held-out episode")
    successes = sum(value >= 0.5 for value in success_values)
    safe_successes = sum(
        value >= 0.5 and record.get("constraint_violated", 0.0) < 0.5
        for value, record in zip(success_values, episode_records)
    )
    metric_means = {
        key: float(np.mean([record[key] for record in episode_records if key in record]))
        for key in sorted({key for record in episode_records for key in record})
    }
    payload = {
        "schema_version": 1,
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "benchmark_semantics": "intervention_target_only_v2",
        "observation_contract": checkpoint["observation_contract"],
        "env_id": task["env_id"], "method": task["method"], "condition": args.condition,
        "training_seed": seed, "checkpoint": "best.pt",
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "training_source_sha256": checkpoint.get("source_sha256"),
        "seed_base": args.seed_base, "episodes": len(episode_records),
        "successes": successes, "success_rate": successes / len(episode_records),
        "success_wilson_95": wilson(successes, len(episode_records)),
        "safe_successes": safe_successes,
        "safe_success_rate": safe_successes / len(episode_records),
        "safe_success_wilson_95": wilson(safe_successes, len(episode_records)),
        "metric_means": metric_means, "episode_records": episode_records,
    }
    filename = "heldout_eval.json" if args.condition == "configured" else f"heldout_eval_{args.condition}.json"
    atomic_json(payload, run_dir / filename)
    print(json.dumps({key: value for key, value in payload.items() if key != "episode_records"}, indent=2, sort_keys=True))
    envs.close()


if __name__ == "__main__":
    main()
