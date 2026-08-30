#!/usr/bin/env python3
"""Held-out deterministic evaluation for the restricted-input visual policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_visual_recovery_ppo import (
    VisualAgent, env_kwargs, extract_observation, metric_success,
    observation_contract, privileged_aux_dim, select_task, visual_progress_target,
)
from evaluation_seed import SEED_DERIVATION, heldout_batch_seed


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


VISUAL_PERTURBATIONS = (
    "none", "pixel_shift_right_4", "brightness_70", "warm_color_shift",
)
PROGRESS_HEAD_MODES = ("normal", "zero", "one", "cyclic_shift")
ENVIRONMENT_PROFILES = (
    "nominal", "camera_left_5cm", "camera_high_5cm", "lighting_dim",
    "lighting_warm",
)


def apply_visual_perturbation(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply a frozen deterministic sensor-space evaluation perturbation."""

    if mode == "none":
        return rgb
    if mode == "pixel_shift_right_4":
        image = rgb.permute(0, 3, 1, 2)
        height, width = image.shape[-2:]
        image = F.pad(image, (4, 4, 4, 4), mode="replicate")
        return image[:, :, 4:4 + height, 0:width].permute(0, 2, 3, 1)
    image = rgb.float()
    if mode == "brightness_70":
        image = image * 0.7
    elif mode == "warm_color_shift":
        scale = torch.tensor(
            [1.15, 0.95, 0.80], device=image.device, dtype=image.dtype,
        )
        image = image * scale
    else:
        raise ValueError(f"unknown visual perturbation: {mode}")
    return image.round().clamp(0, 255).to(rgb.dtype)


def deterministic_action_with_progress_mode(
    agent, rgb: torch.Tensor, proprio: torch.Tensor, mode: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the actor with a frozen intervention on its learned progress head."""

    latent = agent.encode(rgb)
    actor_parts = [latent, proprio]
    predicted = None
    if agent.goal_progress_predictor is not None:
        predicted = torch.sigmoid(agent.goal_progress_predictor(latent))
        if mode == "normal":
            effective = predicted
        elif mode == "zero":
            effective = torch.zeros_like(predicted)
        elif mode == "one":
            effective = torch.ones_like(predicted)
        elif mode == "cyclic_shift":
            effective = torch.roll(predicted, shifts=1, dims=0)
        else:
            raise ValueError(f"unknown progress-head mode: {mode}")
        actor_parts.append(effective)
    elif mode != "normal":
        raise ValueError("progress-head intervention requires a learned head")
    mean = agent.actor(torch.cat(actor_parts, dim=1))
    return torch.tanh(mean), predicted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_ppo_gate_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=81000000)
    parser.add_argument("--condition", choices=("configured", "nominal", "intervention"), default="configured")
    parser.add_argument("--progress-head-mode", choices=PROGRESS_HEAD_MODES, default="normal")
    parser.add_argument("--visual-perturbation", choices=VISUAL_PERTURBATIONS, default="none")
    parser.add_argument("--environment-profile", choices=ENVIRONMENT_PROFILES, default="nominal")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("visual policy evaluation requires CUDA")
    if args.episodes % args.num_envs:
        raise ValueError("episodes must be divisible by num-envs for exact paired evaluation")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = select_task(config, args.task_index)
    registration_module = None
    if task.get("registration_module"):
        registration_module = importlib.import_module(task["registration_module"])
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
    evaluation_env_id = task["env_id"]
    ood_registration_module = None
    if args.environment_profile != "nominal":
        if task["env_id"] != "LearnedRecovery-v3":
            raise ValueError("physical visual-domain profiles require LearnedRecovery-v3")
        ood_registration_module = importlib.import_module(
            "atr.envs.learned_recovery_v3_ood"
        )
        evaluation_env_id = "LearnedRecovery-v3-OOD"
        kwargs["visual_domain_profile"] = args.environment_profile
    envs = gym.make(evaluation_env_id, num_envs=args.num_envs, reconfiguration_freq=1, **kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=True, record_metrics=True)
    observation, _ = envs.reset(seed=args.seed_base + seed * 100000)
    rgb, proprio, critic_state = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(envs.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic_state.shape[1], action_dim,
        task["asymmetric_critic"], task.get("augmentation_pad", 0), privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")
    if checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("checkpoint lacks the restricted visual observation contract")
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    training_complete_path = run_dir / "TRAINING_COMPLETE.json"
    if not training_complete_path.exists():
        raise FileNotFoundError(
            f"held-out evaluation requires completed training: {training_complete_path}"
        )
    online_protocol_ppo_steps = int(json.loads(
        training_complete_path.read_text(encoding="utf-8")
    )["global_step"])
    bc_path = run_dir / "bc_pretraining.json"
    local_bc_transitions = 0
    if bc_path.exists():
        local_bc_transitions = int(
            json.loads(bc_path.read_text(encoding="utf-8"))["bc_transitions"]
        )
    initialization_path = run_dir / "initialization.json"
    initialization = None
    initialization_ppo_steps = 0
    initialization_protocol_ppo_steps = 0
    initialization_bc_transitions = 0
    if initialization_path.exists():
        initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
        initialization_ppo_steps = int(initialization["source_global_step"])
        source_checkpoint_path = Path(initialization["checkpoint"])
        if not source_checkpoint_path.exists():
            raise FileNotFoundError(
                f"initializer checkpoint unavailable during evaluation: {source_checkpoint_path}"
            )
        source_checkpoint = torch.load(
            source_checkpoint_path, map_location="cpu", weights_only=False,
        )
        source_bc_path = source_checkpoint_path.parent / "bc_pretraining.json"
        initialization_bc_transitions = (
            int(json.loads(source_bc_path.read_text(encoding="utf-8"))["bc_transitions"])
            if source_bc_path.exists() else 0
        )
        initialization["source_bc_dagger_environment_transitions"] = (
            initialization_bc_transitions
        )
        initialization["source_observation_contract"] = source_checkpoint.get(
            "observation_contract"
        )
        initialization["source_sha256"] = source_checkpoint.get("source_sha256")
        source_complete_path = source_checkpoint_path.parent / "TRAINING_COMPLETE.json"
        if not source_complete_path.exists():
            raise FileNotFoundError(
                f"initializer lacks completed-training marker: {source_complete_path}"
            )
        initialization_protocol_ppo_steps = int(json.loads(
            source_complete_path.read_text(encoding="utf-8")
        )["global_step"])
        initialization["source_protocol_ppo_environment_steps"] = (
            initialization_protocol_ppo_steps
        )
    online_ppo_steps = int(checkpoint["global_step"])
    ppo_steps = online_ppo_steps + initialization_ppo_steps
    protocol_ppo_steps = online_protocol_ppo_steps + initialization_protocol_ppo_steps
    bc_transitions = local_bc_transitions + initialization_bc_transitions

    completed = 0
    episode_records = []
    progress_correct_bits = 0
    progress_total_bits = 0
    progress_correct_vectors = 0
    progress_total_vectors = 0
    progress_true_positive = 0
    progress_true_negative = 0
    progress_false_positive = 0
    progress_false_negative = 0
    tracked_maxima = (
        "goals_completed", "goals_unavailable", "constraint_violated",
        "intervention_occurred",
    )
    branches = ("first_goal_removed", "instruction_red_first")
    batch_seeds = []
    seen_batch_seeds = set()
    with torch.no_grad():
        while completed < args.episodes:
            # The completed offset and training seed are common across methods,
            # making episode records exactly pairable within each training seed.
            batch_seed = heldout_batch_seed(args.seed_base, seed, completed)
            if batch_seed in seen_batch_seeds:
                raise RuntimeError("held-out batch-seed collision")
            seen_batch_seeds.add(batch_seed)
            batch_seeds.append(batch_seed)
            observation, _ = envs.reset(seed=batch_seed)
            metrics = defaultdict(list)
            maxima = {key: torch.zeros(args.num_envs, device="cuda") for key in tracked_maxima}
            branch_values = {}
            for step in range(int(task["num_eval_steps"])):
                rgb, proprio, _ = extract_observation(
                    observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                    task.get("actor_goal_progress", False),
                )
                rgb = apply_visual_perturbation(rgb, args.visual_perturbation)
                action, predicted_progress_probability = (
                    deterministic_action_with_progress_mode(
                        agent, rgb, proprio, args.progress_head_mode,
                    )
                )
                if predicted_progress_probability is not None:
                    predicted_progress = predicted_progress_probability >= 0.5
                    target_progress = visual_progress_target(observation).bool()
                    matches = predicted_progress == target_progress
                    progress_correct_bits += int(matches.sum())
                    progress_total_bits += int(matches.numel())
                    progress_correct_vectors += int(matches.all(dim=1).sum())
                    progress_total_vectors += int(matches.shape[0])
                    progress_true_positive += int(
                        (predicted_progress & target_progress).sum()
                    )
                    progress_true_negative += int(
                        (~predicted_progress & ~target_progress).sum()
                    )
                    progress_false_positive += int(
                        (predicted_progress & ~target_progress).sum()
                    )
                    progress_false_negative += int(
                        (~predicted_progress & target_progress).sum()
                    )
                observation, _, _, _, info = envs.step(action)
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
        "benchmark_semantics": (
            "event_reward_intervention_target_only_v3"
            if task["env_id"] == "LearnedRecovery-v3"
            else "intervention_target_only_v2"
        ),
        "observation_contract": checkpoint["observation_contract"],
        "env_id": task["env_id"], "evaluation_env_id": evaluation_env_id,
        "environment_profile": args.environment_profile,
        "method": task["method"], "condition": args.condition,
        "progress_head_mode": args.progress_head_mode,
        "visual_perturbation": args.visual_perturbation,
        "training_seed": seed, "checkpoint": "best.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "config": args.config,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "checkpoint_global_step": online_ppo_steps,
        "online_ppo_environment_steps": online_ppo_steps,
        "initialization_ppo_environment_steps": initialization_ppo_steps,
        "ppo_environment_steps": ppo_steps,
        "online_protocol_ppo_environment_steps": online_protocol_ppo_steps,
        "initialization_protocol_ppo_environment_steps": (
            initialization_protocol_ppo_steps
        ),
        "protocol_ppo_environment_steps": protocol_ppo_steps,
        "local_bc_dagger_environment_transitions": local_bc_transitions,
        "initialization_bc_dagger_environment_transitions": initialization_bc_transitions,
        "bc_dagger_environment_transitions": bc_transitions,
        "total_environment_transitions": ppo_steps + bc_transitions,
        "protocol_environment_transitions_consumed": (
            protocol_ppo_steps + bc_transitions
        ),
        "initialization_provenance": initialization,
        "training_source_sha256": checkpoint.get("source_sha256"),
        "evaluation_source_sha256": {
            "evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_seed": hashlib.sha256(
                (Path(__file__).parent / "evaluation_seed.py").read_bytes()
            ).hexdigest(),
            "environment_registration": (
                hashlib.sha256(Path(registration_module.__file__).read_bytes()).hexdigest()
                if registration_module is not None and registration_module.__file__
                else None
            ),
            "ood_environment_registration": (
                hashlib.sha256(Path(ood_registration_module.__file__).read_bytes()).hexdigest()
                if ood_registration_module is not None and ood_registration_module.__file__
                else None
            ),
        },
        "seed_base": args.seed_base,
        "seed_derivation": SEED_DERIVATION,
        "batch_seeds": batch_seeds,
        "episodes": len(episode_records),
        "successes": successes, "success_rate": successes / len(episode_records),
        "success_wilson_95": wilson(successes, len(episode_records)),
        "safe_successes": safe_successes,
        "safe_success_rate": safe_successes / len(episode_records),
        "safe_success_wilson_95": wilson(safe_successes, len(episode_records)),
        "metric_means": metric_means, "episode_records": episode_records,
        "ablation_claim_boundary": (
            "Progress-head interventions test causal dependence of the frozen actor; "
            "sensor-space perturbations test deterministic robustness; environment "
            "profiles separately test rendered camera/lighting changes. Neither is "
            "a real-robot robustness claim."
        ),
    }
    if progress_total_bits:
        target_positives = progress_true_positive + progress_false_negative
        target_negatives = progress_true_negative + progress_false_positive
        predicted_positives = progress_true_positive + progress_false_positive
        positive_recall = (
            progress_true_positive / target_positives if target_positives else None
        )
        negative_recall = (
            progress_true_negative / target_negatives if target_negatives else None
        )
        payload["visual_progress_bit_accuracy"] = progress_correct_bits / progress_total_bits
        payload["visual_progress_exact_accuracy"] = (
            progress_correct_vectors / progress_total_vectors
        )
        payload["visual_progress_balanced_accuracy"] = (
            0.5 * (positive_recall + negative_recall)
            if positive_recall is not None and negative_recall is not None
            else None
        )
        payload["visual_progress_positive_recall"] = positive_recall
        payload["visual_progress_negative_recall"] = negative_recall
        payload["visual_progress_target_positive_rate"] = (
            target_positives / progress_total_bits
        )
        payload["visual_progress_predicted_positive_rate"] = (
            predicted_positives / progress_total_bits
        )
        payload["visual_progress_counts"] = {
            "correct_bits": progress_correct_bits,
            "total_bits": progress_total_bits,
            "correct_vectors": progress_correct_vectors,
            "total_vectors": progress_total_vectors,
            "true_positive": progress_true_positive,
            "true_negative": progress_true_negative,
            "false_positive": progress_false_positive,
            "false_negative": progress_false_negative,
        }
    suffix = []
    if args.condition != "configured":
        suffix.append(args.condition)
    if args.progress_head_mode != "normal":
        suffix.append(f"progress_{args.progress_head_mode}")
    if args.visual_perturbation != "none":
        suffix.append(f"visual_{args.visual_perturbation}")
    if args.environment_profile != "nominal":
        suffix.append(f"env_{args.environment_profile}")
    filename = "heldout_eval" + ("_" + "_".join(suffix) if suffix else "") + ".json"
    atomic_json(payload, run_dir / filename)
    print(json.dumps({key: value for key, value in payload.items() if key != "episode_records"}, indent=2, sort_keys=True))
    envs.close()


if __name__ == "__main__":
    main()
