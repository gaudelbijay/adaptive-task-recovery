#!/usr/bin/env python3
"""Distill V19 across paired rendered domains and explicit sensor shifts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_v19_robust_distillation import action_from_latent, atomic_save
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    select_task,
    visual_progress_target,
)


def make_vector_env(task: dict, count: int, profile: str | None):
    kwargs = env_kwargs(task)
    env_id = task["env_id"]
    if profile is not None:
        env_id = "LearnedRecovery-v3-OOD"
        kwargs["visual_domain_profile"] = profile
    env = gym.make(env_id, num_envs=count, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if not task.get("ignore_terminations_during_pairing", False):
        raise ValueError("V29 requires explicit paired resets with autoreset disabled")
    return ManiSkillVectorEnv(
        env, count, ignore_terminations=True, record_metrics=True,
    )


def paired_state_error(reference: dict, shifted: dict, task: dict) -> float:
    _, reference_proprio, reference_critic = extract_observation(
        reference, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    _, shifted_proprio, shifted_critic = extract_observation(
        shifted, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    return max(
        float((reference_proprio - shifted_proprio).abs().max()),
        float((reference_critic - shifted_critic).abs().max()),
    )


def apply_sensor_shift(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply only the frozen V29 development transforms."""

    if mode == "pixel_shift_right_4":
        image = rgb.permute(0, 3, 1, 2)
        height, width = image.shape[-2:]
        image = F.pad(image, (4, 4, 4, 4), mode="replicate")
        return image[:, :, 4:4 + height, 0:width].permute(0, 2, 3, 1)
    image = rgb.float()
    if mode == "brightness_70":
        image = image * 0.70
    elif mode == "warm_color_shift":
        scale = torch.tensor(
            [1.15, 0.95, 0.80], device=image.device, dtype=image.dtype,
        )
        image = image * scale
    else:
        raise ValueError(f"unknown V29 sensor augmentation: {mode}")
    return image.round().clamp(0, 255).to(rgb.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--task-index", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("rendered-domain distillation requires CUDA")
    importlib.import_module(task["registration_module"])
    ood_module = importlib.import_module("atr.envs.learned_recovery_v3_ood")
    profiles = list(task["render_profiles"])
    sensor_augmentations = list(task["sensor_augmentations"])
    if not profiles or len(profiles) != len(set(profiles)):
        raise ValueError("render profiles must be unique and non-empty")
    if any(profile not in ood_module.PROFILES for profile in profiles):
        raise ValueError("render-distillation profile is not registered")
    expected_sensor_augmentations = {
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    }
    if set(sensor_augmentations) != expected_sensor_augmentations:
        raise ValueError("V29 requires the complete frozen sensor augmentation set")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite rendered-domain run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs_per_profile"])
    segment_steps = int(task["paired_segment_steps"])
    if not 1 <= segment_steps < int(task["env_kwargs"]["intervention_steps"]):
        raise ValueError(
            "paired_segment_steps must be positive and end before the intervention boundary"
        )
    pairs = []
    for profile_index, profile in enumerate(profiles):
        reference_env = make_vector_env(task, count, None)
        shifted_env = make_vector_env(task, count, profile)
        pair_seed = seed + 1_000_003 * (profile_index + 1)
        reference_observation, _ = reference_env.reset(seed=pair_seed)
        shifted_observation, _ = shifted_env.reset(seed=pair_seed)
        error = paired_state_error(reference_observation, shifted_observation, task)
        if error > float(task["maximum_paired_state_error"]):
            raise ValueError(f"paired reset state mismatch for {profile}: {error}")
        pairs.append({
            "profile": profile,
            "reference_env": reference_env,
            "shifted_env": shifted_env,
            "reference_observation": reference_observation,
            "shifted_observation": shifted_observation,
            "pair_seed": pair_seed,
            "reset_index": 0,
        })

    sample = pairs[0]["reference_observation"]
    _, proprio, critic = extract_observation(
        sample, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(pairs[0]["reference_env"].single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    teacher = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    checkpoint_path = Path(str(task["distillation_teacher_checkpoint"]).format(seed=seed))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    expected_contract = observation_contract(task)
    if checkpoint.get("observation_contract") != expected_contract:
        raise ValueError("V19 teacher observation contract mismatch")
    agent.load_state_dict(checkpoint["agent"], strict=True)
    teacher.load_state_dict(checkpoint["agent"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if not task.get("freeze_policy_heads", False):
        raise ValueError("V29 requires frozen policy heads")
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), eps=1e-5,
    )

    updates = int(task["distillation_updates"])
    rollout_max = float(task["student_rollout_max"])
    maximum_error = float(task["maximum_paired_state_error"])
    history = []
    agent.train()
    for update in range(updates):
        per_profile = []
        for pair in pairs:
            reference = pair["reference_observation"]
            shifted = pair["shifted_observation"]
            state_error = paired_state_error(reference, shifted, task)
            if state_error > maximum_error:
                raise ValueError(
                    f"paired state diverged for {pair['profile']} at update {update}: "
                    f"{state_error}"
                )
            reference_rgb, reference_proprio, _ = extract_observation(
                reference, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            shifted_rgb, shifted_proprio, _ = extract_observation(
                shifted, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            with torch.no_grad():
                teacher_reference_latent = teacher.encode(reference_rgb)
                target_action = action_from_latent(
                    teacher, teacher_reference_latent, reference_proprio,
                )
                progress_target = visual_progress_target(shifted)
            reference_latent = agent.encode(reference_rgb)
            shifted_latent = agent.encode(shifted_rgb)
            reference_action = action_from_latent(agent, reference_latent, reference_proprio)
            shifted_action = action_from_latent(agent, shifted_latent, shifted_proprio)
            reference_action_loss = F.mse_loss(reference_action, target_action)
            shifted_action_loss = F.mse_loss(shifted_action, target_action)
            progress_loss = reference_action.new_zeros(())
            if agent.goal_progress_predictor is not None:
                progress_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        agent.goal_progress_predictor(reference_latent), progress_target,
                    )
                    + F.binary_cross_entropy_with_logits(
                        agent.goal_progress_predictor(shifted_latent), progress_target,
                    )
                )
            consistency_loss = 1.0 - F.cosine_similarity(
                shifted_latent, reference_latent.detach(), dim=1,
            ).mean()
            augmented_action_losses = []
            augmented_feature_losses = []
            for mode in sensor_augmentations:
                augmented_latent = agent.encode(apply_sensor_shift(reference_rgb, mode))
                augmented_action = action_from_latent(
                    agent, augmented_latent, reference_proprio,
                )
                augmented_action_losses.append(F.mse_loss(augmented_action, target_action))
                augmented_feature_losses.append(F.mse_loss(
                    augmented_latent, teacher_reference_latent,
                ))
            sensor_action_loss = torch.stack(augmented_action_losses).mean()
            feature_anchor_loss = torch.stack([
                F.mse_loss(reference_latent, teacher_reference_latent),
                F.mse_loss(shifted_latent, teacher_reference_latent),
                *augmented_feature_losses,
            ]).mean()
            loss = (
                float(task["original_action_weight"]) * reference_action_loss
                + float(task["rendered_action_weight"]) * shifted_action_loss
                + float(task["sensor_action_weight"]) * sensor_action_loss
                + float(task["feature_anchor_weight"]) * feature_anchor_loss
                + float(task["progress_weight"]) * progress_loss
                + float(task["latent_consistency_weight"]) * consistency_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                student_probability = rollout_max * update / max(updates - 1, 1)
                student_mask = torch.rand(
                    (target_action.shape[0], 1), device=device,
                ) < student_probability
                executed = torch.where(student_mask, shifted_action.detach(), target_action)
                reference_next, _, _, _, _ = pair["reference_env"].step(executed)
                shifted_next, _, _, _, _ = pair["shifted_env"].step(executed)
                if (update + 1) % segment_steps == 0:
                    pair["reset_index"] += 1
                    reset_seed = pair["pair_seed"] + pair["reset_index"]
                    reference_next, _ = pair["reference_env"].reset(seed=reset_seed)
                    shifted_next, _ = pair["shifted_env"].reset(seed=reset_seed)
                    reset_error = paired_state_error(reference_next, shifted_next, task)
                    if reset_error > maximum_error:
                        raise ValueError(
                            f"paired reset state mismatch for {pair['profile']} "
                            f"at update {update}: {reset_error}"
                        )
            pair["reference_observation"] = reference_next
            pair["shifted_observation"] = shifted_next
            per_profile.append({
                "profile": pair["profile"],
                "loss": float(loss.detach()),
                "original_action_loss": float(reference_action_loss.detach()),
                "rendered_action_loss": float(shifted_action_loss.detach()),
                "sensor_action_loss": float(sensor_action_loss.detach()),
                "feature_anchor_loss": float(feature_anchor_loss.detach()),
                "progress_loss": float(progress_loss.detach()),
                "state_error": state_error,
            })
        history.append(per_profile)
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = [item for group in history[-config["log_freq"]:] for item in group]
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "update": update + 1,
                    "student_transitions": (update + 1) * count * len(profiles),
                    "simulator_transitions": 2 * (update + 1) * count * len(profiles),
                    "loss": float(np.mean([item["loss"] for item in recent])),
                    "original_action_loss": float(np.mean([
                        item["original_action_loss"] for item in recent
                    ])),
                    "rendered_action_loss": float(np.mean([
                        item["rendered_action_loss"] for item in recent
                    ])),
                    "sensor_action_loss": float(np.mean([
                        item["sensor_action_loss"] for item in recent
                    ])),
                    "feature_anchor_loss": float(np.mean([
                        item["feature_anchor_loss"] for item in recent
                    ])),
                    "progress_loss": float(np.mean([
                        item["progress_loss"] for item in recent
                    ])),
                    "maximum_paired_state_error": max(item["state_error"] for item in recent),
                }) + "\n")

    for pair in pairs:
        pair["reference_env"].close()
        pair["shifted_env"].close()
    recent = [item for group in history[-100:] for item in group]
    student_transitions = updates * count * len(profiles)
    simulator_transitions = 2 * student_transitions
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(
            Path(importlib.import_module(task["registration_module"]).__file__).read_bytes()
        ).hexdigest(),
        "rendered_environment": hashlib.sha256(Path(ood_module.__file__).read_bytes()).hexdigest(),
    }
    best_metrics = {
        "mean_last_100_loss": float(np.mean([item["loss"] for item in recent])),
        "mean_last_100_original_action_loss": float(np.mean([
            item["original_action_loss"] for item in recent
        ])),
        "mean_last_100_rendered_action_loss": float(np.mean([
            item["rendered_action_loss"] for item in recent
        ])),
        "mean_last_100_sensor_action_loss": float(np.mean([
            item["sensor_action_loss"] for item in recent
        ])),
        "mean_last_100_feature_anchor_loss": float(np.mean([
            item["feature_anchor_loss"] for item in recent
        ])),
        "mean_last_100_progress_loss": float(np.mean([
            item["progress_loss"] for item in recent
        ])),
        "maximum_last_100_paired_state_error": max(item["state_error"] for item in recent),
    }
    payload = {
        "schema_version": 1,
        "training_protocol": "paired_rendered_sensor_distillation",
        "observation_contract": expected_contract,
        "source_sha256": source_hashes,
        "task": task,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": updates,
        "global_step": student_transitions,
        "best_score": -best_metrics["mean_last_100_loss"],
        "best_metrics": best_metrics,
    }
    atomic_save(payload, run_dir / "best.pt")
    atomic_save(payload, run_dir / "latest.pt")
    completion = {
        "schema_version": 1,
        "training_protocol": "paired_rendered_sensor_distillation",
        "global_step": student_transitions,
        "distillation_updates": updates,
        "student_transitions": student_transitions,
        "simulator_transitions": simulator_transitions,
        "ppo_environment_steps": 0,
        "teacher_checkpoint": str(checkpoint_path),
        "teacher_checkpoint_sha256": file_sha256(checkpoint_path),
        "render_profiles": profiles,
        "sensor_augmentations": sensor_augmentations,
        "freeze_policy_heads": True,
        "paired_segment_steps": segment_steps,
        **best_metrics,
        "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
