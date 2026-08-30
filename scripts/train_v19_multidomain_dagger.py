#!/usr/bin/env python3
"""Full-episode multidomain DAgger from V19 and privileged state teachers."""

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

from train_manipulation_ppo import Agent as StateAgent
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent,
    atomic_save,
    dual_teacher_strict_route,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    reconstruct_state_teacher_observation,
    select_task,
    visual_progress_target,
)


def make_vector_env(task: dict, count: int, profile: str):
    kwargs = env_kwargs(task)
    env_id = task["env_id"]
    if profile != "nominal":
        env_id = "LearnedRecovery-v3-OOD"
        kwargs["visual_domain_profile"] = profile
    env = gym.make(env_id, num_envs=count, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, count, record_metrics=True)


def apply_sensor_shift(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "pixel_shift_right_4":
        image = rgb.permute(0, 3, 1, 2)
        height, width = image.shape[-2:]
        image = F.pad(image, (4, 4, 4, 4), mode="replicate")
        return image[:, :, 4:4 + height, 0:width].permute(0, 2, 3, 1)
    image = rgb.float()
    if mode == "brightness_70":
        image = image * 0.70
    elif mode == "warm_color_shift":
        image = image * torch.tensor(
            [1.15, 0.95, 0.80], device=image.device, dtype=image.dtype,
        )
    else:
        raise ValueError(f"unknown V30 sensor augmentation: {mode}")
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
        raise RuntimeError("multidomain DAgger requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    rendered = importlib.import_module("atr.envs.learned_recovery_v3_ood")
    profiles = list(task["domain_profiles"])
    if profiles[0] != "nominal" or len(profiles) != len(set(profiles)):
        raise ValueError("V30 domains must be unique and start with nominal")
    if any(profile not in rendered.PROFILES for profile in profiles[1:]):
        raise ValueError("V30 contains an unregistered rendered profile")
    sensor_augmentations = list(task["sensor_augmentations"])
    if set(sensor_augmentations) != {
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    }:
        raise ValueError("V30 requires the complete observed sensor-shift set")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V30 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs_per_domain"])
    domains = []
    for index, profile in enumerate(profiles):
        env = make_vector_env(task, count, profile)
        observation, _ = env.reset(seed=seed + 1_000_003 * (index + 1))
        domains.append({"profile": profile, "env": env, "observation": observation})
    sample = domains[0]["observation"]
    _, proprio, critic = extract_observation(
        sample, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(domains[0]["env"].single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_teacher = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    nominal_path = Path(str(task["nominal_state_teacher_checkpoint"]).format(seed=seed))
    strict_path = Path(str(task["strict_state_teacher_checkpoint"]).format(seed=seed))
    for path in (source_path, nominal_path, strict_path):
        if not path.exists():
            raise FileNotFoundError(path)
    source_checkpoint = torch.load(source_path, map_location=device, weights_only=False)
    if source_checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("V30 source visual contract mismatch")
    agent.load_state_dict(source_checkpoint["agent"], strict=True)
    source_teacher.load_state_dict(source_checkpoint["agent"], strict=True)
    state_sample = reconstruct_state_teacher_observation(sample)
    nominal_teacher = StateAgent(state_sample.shape[1], action_dim).to(device)
    strict_teacher = StateAgent(state_sample.shape[1], action_dim).to(device)
    nominal_teacher.load_state_dict(
        torch.load(nominal_path, map_location=device, weights_only=False)["agent"],
        strict=True,
    )
    strict_teacher.load_state_dict(
        torch.load(strict_path, map_location=device, weights_only=False)["agent"],
        strict=True,
    )
    for teacher in (source_teacher, nominal_teacher, strict_teacher):
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(config["learning_rate"]), eps=1e-5)

    updates = int(task["dagger_updates"])
    expected_transitions = updates * count * len(profiles)
    if int(task["total_timesteps"]) != expected_transitions:
        raise ValueError("V30 declared transition budget is inconsistent")
    rollout_max = float(task["student_rollout_max"])
    history = []
    agent.train()
    for update in range(updates):
        per_domain = []
        optimizer.zero_grad()
        for domain in domains:
            observation = domain["observation"]
            rgb, proprio, _ = extract_observation(
                observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            state = reconstruct_state_teacher_observation(observation)
            with torch.no_grad():
                source_action = source_teacher.get_action(rgb, proprio, deterministic=True)
                nominal_action = torch.clamp(
                    nominal_teacher.get_action(state, deterministic=True), -1.0, 1.0,
                )
                strict_action = torch.clamp(
                    strict_teacher.get_action(state, deterministic=True), -1.0, 1.0,
                )
                strict_route = dual_teacher_strict_route(observation)
                routed_state_action = torch.where(strict_route, strict_action, nominal_action)
                teacher_action = (
                    source_action if domain["profile"] == "nominal" else routed_state_action
                )
                progress_target = visual_progress_target(observation)
                source_latent = source_teacher.encode(rgb)
            latent = agent.encode(rgb)
            student_action = agent.get_action(rgb, proprio, deterministic=True)
            action_loss = F.mse_loss(student_action, teacher_action)
            sensor_losses = []
            for mode in sensor_augmentations:
                shifted_action = agent.get_action(
                    apply_sensor_shift(rgb, mode), proprio, deterministic=True,
                )
                sensor_losses.append(F.mse_loss(shifted_action, teacher_action))
            sensor_loss = torch.stack(sensor_losses).mean()
            progress_loss = F.binary_cross_entropy_with_logits(
                agent.goal_progress_predictor(latent), progress_target,
            )
            source_anchor_loss = latent.new_zeros(())
            if domain["profile"] == "nominal":
                source_anchor_loss = F.mse_loss(latent, source_latent)
            loss = (
                float(task["action_weight"]) * action_loss
                + float(task["sensor_action_weight"]) * sensor_loss
                + float(task["progress_weight"]) * progress_loss
                + float(task["source_feature_anchor_weight"]) * source_anchor_loss
            )
            (loss / len(domains)).backward()
            with torch.no_grad():
                student_probability = rollout_max * update / max(updates - 1, 1)
                student_mask = torch.rand((count, 1), device=device) < student_probability
                executed = torch.where(student_mask, student_action.detach(), teacher_action)
                next_observation, _, _, _, _ = domain["env"].step(executed)
            domain["observation"] = next_observation
            per_domain.append({
                "profile": domain["profile"], "loss": float(loss.detach()),
                "action_loss": float(action_loss.detach()),
                "sensor_action_loss": float(sensor_loss.detach()),
                "progress_loss": float(progress_loss.detach()),
                "source_feature_anchor_loss": float(source_anchor_loss.detach()),
                "strict_route_fraction": float(strict_route.float().mean()),
                "student_rollout_fraction": float(student_mask.float().mean()),
            })
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        history.append(per_domain)
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = [item for group in history[-int(config["log_freq"]):] for item in group]
            payload = {
                "update": update + 1,
                "student_transitions": (update + 1) * count * len(profiles),
            }
            for key in (
                "loss", "action_loss", "sensor_action_loss", "progress_loss",
                "source_feature_anchor_loss", "strict_route_fraction",
                "student_rollout_fraction",
            ):
                payload[key] = float(np.mean([item[key] for item in recent]))
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    for domain in domains:
        domain["env"].close()
    recent = [item for group in history[-100:] for item in group]
    transitions = expected_transitions
    best_metrics = {
        f"mean_last_100_{key}": float(np.mean([item[key] for item in recent]))
        for key in (
            "loss", "action_loss", "sensor_action_loss", "progress_loss",
            "source_feature_anchor_loss", "strict_route_fraction",
            "student_rollout_fraction",
        )
    }
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
        "rendered_environment": hashlib.sha256(Path(rendered.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1,
        "training_protocol": "full_episode_multidomain_dagger",
        "observation_contract": observation_contract(task),
        "source_sha256": source_hashes,
        "task": task,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": updates,
        "global_step": transitions,
        "best_score": -best_metrics["mean_last_100_loss"],
        "best_metrics": best_metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    completion = {
        "schema_version": 1,
        "training_protocol": "full_episode_multidomain_dagger",
        "global_step": transitions,
        "dagger_updates": updates,
        "dagger_environment_transitions": transitions,
        "student_transitions": transitions,
        "simulator_transitions": transitions,
        "ppo_environment_steps": 0,
        "domain_profiles": profiles,
        "sensor_augmentations": sensor_augmentations,
        "source_visual_checkpoint": str(source_path),
        "source_visual_checkpoint_sha256": file_sha256(source_path),
        "nominal_state_teacher_checkpoint": str(nominal_path),
        "nominal_state_teacher_checkpoint_sha256": file_sha256(nominal_path),
        "strict_state_teacher_checkpoint": str(strict_path),
        "strict_state_teacher_checkpoint_sha256": file_sha256(strict_path),
        **best_metrics,
        "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
