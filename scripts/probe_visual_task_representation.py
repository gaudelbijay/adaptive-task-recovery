#!/usr/bin/env python3
"""Held-out linear probe of task-progress semantics from visual latents."""

from __future__ import annotations

import argparse
import hashlib
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

from train_visual_recovery_ppo import (
    VisualAgent, env_kwargs, extract_observation, observation_contract,
    privileged_aux_dim, select_task,
)


TARGETS = ("red_goal_resolved", "blue_goal_resolved")


def collect(agent, behavior_agent, envs, task, behavior_task, samples, seed):
    # Environment randomization currently draws from process-global Torch RNGs.
    # Reset them explicitly so different evaluated architectures cannot perturb
    # the supposedly matched trajectory through initialization-side RNG use.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    observation, _ = envs.reset(seed=seed)
    features, targets, images = [], [], []
    with torch.no_grad():
        while sum(item.shape[0] for item in features) < samples:
            rgb, proprio, _ = extract_observation(
                observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            features.append(agent.encode(rgb).cpu())
            images.append(rgb.cpu())
            targets.append(
                observation["extra"]["critic_goal_resolved"].float().cpu()
            )
            behavior_rgb, behavior_proprio, _ = extract_observation(
                observation, behavior_task["asymmetric_critic"],
                behavior_task.get("actor_tcp_pose", False),
                behavior_task.get("actor_goal_progress", False),
            )
            observation, _, _, _, _ = envs.step(
                behavior_agent.get_action(behavior_rgb, behavior_proprio, True)
            )
    return (
        torch.cat(features)[:samples].double(),
        torch.cat(targets)[:samples].double(),
        torch.cat(images)[:samples],
    )


def dataset_sha256(images, targets):
    """Hash exact probe pixels and labels without lossy serialization."""
    digest = hashlib.sha256()
    for tensor in (images, targets):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def ridge_prediction(train_x, train_y, test_x, regularization):
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
    return (test_x @ weights) * y_std + y_mean


def binary_auc(scores, labels):
    labels = labels.bool()
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        raise ValueError("semantic probe target lacks both held-out classes")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.double)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[labels].sum()
    return float(
        (positive_rank_sum - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def binary_probe(train_x, train_y, test_x, test_y, regularization):
    prediction = ridge_prediction(
        train_x, train_y, test_x, regularization,
    )
    error = prediction - test_y
    baseline_error = train_y.mean(0) - test_y
    sse = error.square().sum()
    sst = (test_y - test_y.mean(0)).square().sum().clamp_min(1e-12)
    per_target_sst = (test_y - test_y.mean(0)).square().sum(0)
    per_target_r2 = torch.where(
        per_target_sst > 1e-12,
        1 - error.square().sum(0) / per_target_sst,
        torch.full_like(per_target_sst, float("nan")),
    )
    predicted = prediction >= 0.5
    balanced_accuracies = []
    aucs = []
    prevalences = []
    for index in range(test_y.shape[1]):
        truth = test_y[:, index] >= 0.5
        positive = truth.sum().clamp_min(1)
        negative = (~truth).sum().clamp_min(1)
        true_positive_rate = (predicted[:, index] & truth).sum() / positive
        true_negative_rate = ((~predicted[:, index]) & (~truth)).sum() / negative
        balanced_accuracies.append(float((true_positive_rate + true_negative_rate) / 2))
        aucs.append(binary_auc(prediction[:, index], truth))
        prevalences.append(float(truth.float().mean()))
    return {
        "r2_variance_weighted": float(1 - sse / sst),
        "macro_balanced_accuracy": float(np.mean(balanced_accuracies)),
        "macro_roc_auc": float(np.mean(aucs)),
        "per_target_balanced_accuracy": balanced_accuracies,
        "per_target_roc_auc": aucs,
        "per_target_positive_prevalence": prevalences,
        "mean_absolute_error": float(error.abs().mean()),
        "mean_baseline_absolute_error": float(baseline_error.abs().mean()),
        "per_target_r2": [
            float(value) if torch.isfinite(value) else None for value in per_target_r2
        ],
        "per_target_mae": [float(value) for value in error.abs().mean(0)],
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
    registration_module = None
    if task.get("registration_module"):
        registration_module = importlib.import_module(task["registration_module"])
    seed = int(task["seed"])
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("checkpoint lacks restricted visual contract")

    kwargs = env_kwargs(task, evaluation=True)
    kwargs["asymmetric_critic_observation"] = True
    kwargs["intervention_probability"] = 1.0
    envs = gym.make(task["env_id"], num_envs=args.num_envs, reconfiguration_freq=1, **kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=False)
    observation, _ = envs.reset(seed=args.seed_base + seed)
    rgb, proprio, privileged = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], privileged.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    agent.load_state_dict(checkpoint["agent"]); agent.eval()

    behavior_pattern = config.get("representation_probe_behavior_checkpoint")
    if not behavior_pattern:
        raise ValueError(
            "representation probe requires a frozen behavior checkpoint so "
            "methods are evaluated on a matched RGB distribution"
        )
    behavior_path = Path(str(behavior_pattern).format(seed=seed))
    if not behavior_path.exists():
        raise FileNotFoundError(f"probe behavior checkpoint unavailable: {behavior_path}")
    behavior_checkpoint = torch.load(
        behavior_path, map_location="cuda", weights_only=False,
    )
    behavior_task = behavior_checkpoint["task"]
    for key in ("env_id", "control_mode", "image_size"):
        if behavior_task.get(key) != task.get(key):
            raise ValueError(f"probe behavior {key} does not match evaluated policy")
    behavior_rgb, behavior_proprio, behavior_privileged = extract_observation(
        observation, behavior_task["asymmetric_critic"],
        behavior_task.get("actor_tcp_pose", False),
        behavior_task.get("actor_goal_progress", False),
    )
    behavior_agent = VisualAgent(
        behavior_task["image_size"], behavior_proprio.shape[1],
        behavior_privileged.shape[1], action_dim,
        behavior_task["asymmetric_critic"], 0, privileged_aux_dim(behavior_task),
        behavior_task.get("actor_learned_goal_progress", False),
    ).cuda()
    if behavior_checkpoint.get("observation_contract") != observation_contract(
        behavior_task
    ):
        raise ValueError("probe behavior checkpoint has an invalid observation contract")
    behavior_agent.load_state_dict(behavior_checkpoint["agent"])
    behavior_agent.eval()

    train_x, train_y, train_images = collect(
        agent, behavior_agent, envs, task, behavior_task, args.samples,
        args.seed_base + seed * 10,
    )
    test_x, test_y, test_images = collect(
        agent, behavior_agent, envs, task, behavior_task, args.samples,
        args.seed_base + seed * 10 + 1,
    )
    learned = binary_probe(train_x, train_y, test_x, test_y, args.ridge)

    # A matched random encoder controls for information recoverable from an
    # untrained high-dimensional convolutional projection.
    torch.manual_seed(seed + 44444)
    random_agent = VisualAgent(
        task["image_size"], proprio.shape[1], privileged.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
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
    random_result = binary_probe(
        random_train_x, train_y, random_test_x, test_y, args.ridge,
    )
    payload = {
        "schema_version": 1,
        "protocol": "held-out linear goal-resolution probe; labels unavailable to actor",
        "benchmark_semantics": (
            "event_reward_intervention_target_only_v3"
            if task["env_id"] == "LearnedRecovery-v3"
            else "intervention_target_only_v2"
        ),
        "observation_contract": checkpoint["observation_contract"],
        "method": task["method"], "training_seed": seed,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "training_source_sha256": checkpoint.get("source_sha256"),
        "probe_dataset": {
            "protocol": "frozen seed-matched behavior policy; identical pixels across methods",
            "behavior_checkpoint": str(behavior_path),
            "behavior_method": behavior_task["method"],
            "behavior_checkpoint_global_step": int(
                behavior_checkpoint["global_step"]
            ),
            "behavior_observation_contract": behavior_checkpoint[
                "observation_contract"
            ],
            "behavior_training_source_sha256": behavior_checkpoint.get(
                "source_sha256"
            ),
            "train_seed": args.seed_base + seed * 10,
            "test_seed": args.seed_base + seed * 10 + 1,
            "train_sha256": dataset_sha256(train_images, train_y),
            "test_sha256": dataset_sha256(test_images, test_y),
        },
        "probe_source_sha256": {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment_registration": (
                hashlib.sha256(Path(registration_module.__file__).read_bytes()).hexdigest()
                if registration_module is not None and registration_module.__file__
                else None
            ),
        },
        "train_samples": args.samples, "test_samples": args.samples,
        "ridge_regularization": args.ridge,
        "targets": list(TARGETS),
        "learned_encoder": learned, "random_encoder": random_result,
        "learned_minus_random_r2": learned["r2_variance_weighted"] - random_result["r2_variance_weighted"],
        "learned_minus_random_balanced_accuracy": (
            learned["macro_balanced_accuracy"]
            - random_result["macro_balanced_accuracy"]
        ),
        "learned_minus_random_roc_auc": (
            learned["macro_roc_auc"] - random_result["macro_roc_auc"]
        ),
    }
    atomic_json(payload, run_dir / "task_representation_probe.json")
    print(json.dumps(payload, indent=2, sort_keys=True))
    envs.close()


if __name__ == "__main__":
    main()
