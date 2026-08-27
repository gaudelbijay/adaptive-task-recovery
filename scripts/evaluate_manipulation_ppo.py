#!/usr/bin/env python3
"""Independent deterministic evaluation of a trained manipulation policy.

Evaluation seeds are deliberately disjoint from both the training seed and the
training-time evaluation stream.  Each vector slot is reconfigured for each
batch, giving a held-out set of object/goal configurations rather than merely
replaying the checkpoint-selection episodes.
"""

from __future__ import annotations

import argparse
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

from train_manipulation_ppo import Agent, _environment_kwargs, _metric_success, _select_task


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [float("nan"), float("nan")]
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=81000000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("manipulation policy evaluation requires a CUDA GPU")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = _select_task(config, args.task_index)
    seed = int(task["seed"])
    run_dir = Path(args.output) / config["name"] / task["env_id"] / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"best checkpoint is not available: {checkpoint_path}")

    device = torch.device("cuda")
    env_kwargs = _environment_kwargs(task)
    envs = gym.make(
        task["env_id"], num_envs=args.num_envs, reconfiguration_freq=1, **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    observation_dim = int(np.prod(envs.single_observation_space.shape))
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(observation_dim, action_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    completed = 0
    episode_records: list[dict] = []
    max_steps = int(task["num_eval_steps"])
    with torch.no_grad():
        while completed < args.episodes:
            batch_seed = args.seed_base + args.task_index * 100000 + completed
            observation, _ = envs.reset(seed=batch_seed)
            metrics = defaultdict(list)
            for _ in range(max_steps):
                observation, _, _, _, info = envs.step(agent.get_action(observation, deterministic=True))
                if "final_info" in info:
                    mask = info["_final_info"]
                    for key, value in info["final_info"]["episode"].items():
                        metrics[key].extend(value[mask].detach().float().cpu().tolist())
            available = max((len(values) for values in metrics.values()), default=0)
            take = min(available, args.episodes - completed)
            if take == 0:
                raise RuntimeError(f"no completed episodes after {max_steps} evaluation steps")
            for index in range(take):
                episode_records.append({key: float(values[index]) for key, values in metrics.items()})
            completed += take

    success_values = []
    for record in episode_records:
        value = _metric_success(record)
        if not math.isnan(value):
            success_values.append(value)
    successes = int(sum(value >= 0.5 for value in success_values))
    if len(success_values) != len(episode_records):
        raise RuntimeError(
            "success metric was missing for "
            f"{len(episode_records) - len(success_values)} held-out episodes"
        )
    metric_means = {
        key: float(np.mean([record[key] for record in episode_records if key in record]))
        for key in sorted({key for record in episode_records for key in record})
    }
    payload = {
        "schema_version": 1,
        "protocol": "held-out deterministic state-policy evaluation",
        "env_id": task["env_id"],
        "training_seed": seed,
        "checkpoint": "best.pt",
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "seed_base": args.seed_base,
        "episodes": len(episode_records),
        "success_trials": len(success_values),
        "successes": successes,
        "success_rate": successes / len(success_values),
        "success_wilson_95": _wilson(successes, len(success_values)),
        "metric_means": metric_means,
    }
    _atomic_json(payload, run_dir / "heldout_eval.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    envs.close()


if __name__ == "__main__":
    main()
