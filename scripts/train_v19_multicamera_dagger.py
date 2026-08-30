#!/usr/bin/env python3
"""Full-episode V19 DAgger over simultaneous same-physics camera views."""

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

from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent,
    atomic_save,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    select_task,
    visual_progress_target,
)


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
        raise ValueError(f"unknown V31 sensor augmentation: {mode}")
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
        raise RuntimeError("multicamera DAgger requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    canonical_environment = importlib.import_module("atr.envs.learned_recovery_v3")
    camera_keys = list(task["camera_keys"])
    if camera_keys != ["base_camera", "camera_left_5cm", "camera_high_5cm"]:
        raise ValueError("V31 requires the frozen three-camera ordering")
    augmentations = list(task["sensor_augmentations"])
    if set(augmentations) != {
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    }:
        raise ValueError("V31 requires the complete observed sensor-shift set")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V31 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    kwargs = env_kwargs(task)
    env = gym.make(task["training_env_id"], num_envs=count, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, count, record_metrics=True)
    observation, _ = env.reset(seed=seed)
    if set(observation["sensor_data"]) != set(camera_keys):
        raise ValueError("V31 multicamera observation keys mismatch")
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
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
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V31 source visual contract mismatch")
    agent.load_state_dict(source["agent"], strict=True)
    teacher.load_state_dict(source["agent"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.parameters(), lr=float(config["learning_rate"]), eps=1e-5)

    updates = int(task["dagger_updates"])
    expected_transitions = updates * count
    if int(task["total_timesteps"]) != expected_transitions:
        raise ValueError("V31 declared transition budget is inconsistent")
    rollout_max = float(task["student_rollout_max"])
    history = []
    agent.train()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {
            key: observation["sensor_data"][key]["rgb"] for key in camera_keys
        }
        base_rgb = views["base_camera"]
        with torch.no_grad():
            target_action = teacher.get_action(base_rgb, proprio, deterministic=True)
            teacher_latent = teacher.encode(base_rgb)
            progress_target = visual_progress_target(observation)
        latents = {key: agent.encode(rgb) for key, rgb in views.items()}
        actions = {
            key: agent.get_action(rgb, proprio, deterministic=True)
            for key, rgb in views.items()
        }
        source_action_loss = F.mse_loss(actions["base_camera"], target_action)
        camera_action_loss = torch.stack([
            F.mse_loss(actions[key], target_action) for key in camera_keys[1:]
        ]).mean()
        augmentation = augmentations[update % len(augmentations)]
        sensor_action_loss = torch.stack([
            F.mse_loss(
                agent.get_action(apply_sensor_shift(views[key], augmentation), proprio,
                                 deterministic=True),
                target_action,
            )
            for key in camera_keys
        ]).mean()
        progress_loss = torch.stack([
            F.binary_cross_entropy_with_logits(
                agent.goal_progress_predictor(latents[key]), progress_target,
            )
            for key in camera_keys
        ]).mean()
        source_feature_anchor_loss = F.mse_loss(
            latents["base_camera"], teacher_latent,
        )
        loss = (
            float(task["source_action_weight"]) * source_action_loss
            + float(task["camera_action_weight"]) * camera_action_loss
            + float(task["sensor_action_weight"]) * sensor_action_loss
            + float(task["progress_weight"]) * progress_loss
            + float(task["source_feature_anchor_weight"]) * source_feature_anchor_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            student_probability = rollout_max * update / max(updates - 1, 1)
            student_mask = torch.rand((count, 1), device=device) < student_probability
            executed = torch.where(
                student_mask, actions["base_camera"].detach(), target_action,
            )
            observation, _, _, _, _ = env.step(executed)
        history.append({
            "loss": float(loss.detach()),
            "source_action_loss": float(source_action_loss.detach()),
            "camera_action_loss": float(camera_action_loss.detach()),
            "sensor_action_loss": float(sensor_action_loss.detach()),
            "progress_loss": float(progress_loss.detach()),
            "source_feature_anchor_loss": float(source_feature_anchor_loss.detach()),
            "student_rollout_fraction": float(student_mask.float().mean()),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "student_transitions": (update + 1) * count}
            for key in history[-1]:
                payload[key] = float(np.mean([item[key] for item in recent]))
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    env.close()
    recent = history[-100:]
    transitions = expected_transitions
    best_metrics = {
        f"mean_last_100_{key}": float(np.mean([item[key] for item in recent]))
        for key in history[-1]
    }
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(Path(canonical_environment.__file__).read_bytes()).hexdigest(),
        "multicamera_environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1,
        "training_protocol": "full_episode_same_physics_multicamera_dagger",
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
        "training_protocol": "full_episode_same_physics_multicamera_dagger",
        "global_step": transitions,
        "dagger_updates": updates,
        "dagger_environment_transitions": transitions,
        "student_transitions": transitions,
        "simulator_transitions": transitions,
        "ppo_environment_steps": 0,
        "camera_keys": camera_keys,
        "sensor_augmentations": augmentations,
        "source_visual_checkpoint": str(source_path),
        "source_visual_checkpoint_sha256": file_sha256(source_path),
        **best_metrics,
        "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
