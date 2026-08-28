#!/usr/bin/env python3
"""Held-out linear probe of visual latents using analysis-only pose labels."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_visual_recovery_ppo import VisualAgent, env_kwargs, extract_observation, select_task


POSE_KEYS = (
    "critic_red_cube_pose", "critic_blue_cube_pose",
    "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
)


def collect(agent, envs, task, samples, seed):
    observation, _ = envs.reset(seed=seed)
    features, targets, images = [], [], []
    with torch.no_grad():
        while sum(item.shape[0] for item in features) < samples:
            rgb, proprio, _ = extract_observation(observation, task["asymmetric_critic"])
            features.append(agent.encode(rgb).cpu())
            images.append(rgb.cpu())
            targets.append(torch.cat([
                observation["extra"][key][:, :3] for key in POSE_KEYS
            ], dim=1).cpu())
            observation, _, _, _, _ = envs.step(agent.get_action(rgb, proprio, True))
    return (
        torch.cat(features)[:samples].double(),
        torch.cat(targets)[:samples].double(),
        torch.cat(images)[:samples],
    )


def ridge_probe(train_x, train_y, test_x, test_y, regularization):
    x_mean, x_std = train_x.mean(0), train_x.std(0).clamp_min(1e-6)
    y_mean, y_std = train_y.mean(0), train_y.std(0).clamp_min(1e-6)
    train_x = (train_x - x_mean) / x_std
    test_x = (test_x - x_mean) / x_std
    train_y_normalized = (train_y - y_mean) / y_std
    train_x = torch.cat((train_x, torch.ones((len(train_x), 1), dtype=train_x.dtype)), dim=1)
    test_x = torch.cat((test_x, torch.ones((len(test_x), 1), dtype=test_x.dtype)), dim=1)
    identity = torch.eye(train_x.shape[1], dtype=train_x.dtype)
    identity[-1, -1] = 0
    weights = torch.linalg.solve(
        train_x.T @ train_x + regularization * identity,
        train_x.T @ train_y_normalized,
    )
    prediction = (test_x @ weights) * y_std + y_mean
    error = prediction - test_y
    baseline_error = y_mean - test_y
    sse = error.square().sum()
    sst = (test_y - test_y.mean(0)).square().sum().clamp_min(1e-12)
    per_target_sst = (test_y - test_y.mean(0)).square().sum(0)
    per_target_r2 = torch.where(
        per_target_sst > 1e-12,
        1 - error.square().sum(0) / per_target_sst,
        torch.full_like(per_target_sst, float("nan")),
    )
    return {
        "r2_variance_weighted": float(1 - sse / sst),
        "mean_absolute_error_m": float(error.abs().mean()),
        "mean_baseline_absolute_error_m": float(baseline_error.abs().mean()),
        "per_coordinate_r2": [
            float(value) if torch.isfinite(value) else None for value in per_target_r2
        ],
        "per_coordinate_mae_m": [float(value) for value in error.abs().mean(0)],
    }


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_intervention_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=93000000)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("representation probing requires CUDA")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = select_task(config, args.task_index)
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])
    seed = int(task["seed"])
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint.get("observation_contract") != "rgb_qpos_qvel_instruction_v1":
        raise ValueError("checkpoint lacks restricted visual contract")

    kwargs = env_kwargs(task, evaluation=True)
    kwargs["asymmetric_critic_observation"] = True
    kwargs["intervention_probability"] = 1.0
    envs = gym.make(task["env_id"], num_envs=args.num_envs, reconfiguration_freq=1, **kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=False)
    observation, _ = envs.reset(seed=args.seed_base + seed)
    rgb, proprio, privileged = extract_observation(observation, task["asymmetric_critic"])
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], privileged.shape[1], action_dim,
        task["asymmetric_critic"], 0,
    ).cuda()
    agent.load_state_dict(checkpoint["agent"]); agent.eval()

    train_x, train_y, train_images = collect(
        agent, envs, task, args.samples, args.seed_base + seed * 10,
    )
    test_x, test_y, test_images = collect(
        agent, envs, task, args.samples, args.seed_base + seed * 10 + 1,
    )
    learned = ridge_probe(train_x, train_y, test_x, test_y, args.ridge)

    # A matched random encoder controls for information recoverable from an
    # untrained high-dimensional convolutional projection.
    torch.manual_seed(seed + 44444)
    random_agent = VisualAgent(
        task["image_size"], proprio.shape[1], privileged.shape[1], action_dim,
        task["asymmetric_critic"], 0,
    ).cuda().eval()
    batch = 512
    with torch.no_grad():
        random_train_x = torch.cat([
            random_agent.encode(train_images[start:start + batch].cuda()).cpu()
            for start in range(0, len(train_images), batch)
        ]).double()
        random_test_x = torch.cat([
            random_agent.encode(test_images[start:start + batch].cuda()).cpu()
            for start in range(0, len(test_images), batch)
        ]).double()
    random_result = ridge_probe(
        random_train_x, train_y, random_test_x, test_y, args.ridge,
    )
    payload = {
        "schema_version": 1,
        "protocol": "held-out linear pose probe; labels unavailable to actor",
        "benchmark_semantics": "intervention_target_only_v2",
        "observation_contract": checkpoint["observation_contract"],
        "method": task["method"], "training_seed": seed,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "training_source_sha256": checkpoint.get("source_sha256"),
        "train_samples": args.samples, "test_samples": args.samples,
        "targets": [f"{key}:{axis}" for key in POSE_KEYS for axis in ("x", "y", "z")],
        "learned_encoder": learned, "random_encoder": random_result,
        "learned_minus_random_r2": learned["r2_variance_weighted"] - random_result["r2_variance_weighted"],
    }
    atomic_json(payload, run_dir / "representation_probe.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    envs.close()


if __name__ == "__main__":
    main()
