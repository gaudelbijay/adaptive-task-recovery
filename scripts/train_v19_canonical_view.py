#!/usr/bin/env python3
"""Train paired RGB canonical-view synthesis in front of immutable V19."""

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

from train_v19_multicamera_dagger import apply_sensor_shift
from train_visual_recovery_dual_teacher_ppo import (
    atomic_save,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    select_task,
    visual_progress_target,
)
from v33_canonical_view_agent import CanonicalizedV19Agent


def action_from_latent(agent, latent, proprio):
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1)))


def image_gradient_loss(prediction, target):
    pred_x = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_y = prediction[:, 1:, :, :] - prediction[:, :-1, :, :]
    target_y = target[:, 1:, :, :] - target[:, :-1, :, :]
    return F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y)


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
        raise RuntimeError("V33 canonical-view training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    canonical_environment = importlib.import_module("atr.envs.learned_recovery_v3")
    camera_keys = list(task["camera_keys"])
    augmentations = list(task["sensor_augmentations"])
    if camera_keys != ["base_camera", "camera_left_5cm", "camera_high_5cm"]:
        raise ValueError("V33 requires the frozen same-physics camera ordering")
    if augmentations != [
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    ]:
        raise ValueError("V33 requires all observed sensor domains")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V33 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    env = gym.make(task["training_env_id"], num_envs=count, **env_kwargs(task))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, count, record_metrics=True)
    observation, _ = env.reset(seed=seed)
    if set(observation["sensor_data"]) != set(camera_keys):
        raise ValueError("V33 multicamera observation keys mismatch")
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = CanonicalizedV19Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V33 source visual contract mismatch")
    agent.initialize_from_v19(source["agent"])
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(config["learning_rate"]), eps=1e-5)

    updates = int(task["dagger_updates"])
    expected_transitions = updates * count
    if int(task["total_timesteps"]) != expected_transitions:
        raise ValueError("V33 declared transition budget is inconsistent")
    history = []
    agent.train()
    agent.base.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {key: observation["sensor_data"][key]["rgb"] for key in camera_keys}
        base_rgb = views["base_camera"]
        # Every observed sensor domain appears in every update.  V31/V32's
        # cyclic exposure let the router ignore photometric shifts.
        domain_rgbs = [views[camera_keys[1]], views[camera_keys[2]]]
        domain_rgbs.extend(apply_sensor_shift(base_rgb, mode) for mode in augmentations)
        with torch.no_grad():
            target_latent = agent.base.encode(base_rgb)
            target_action = agent.base.get_action(base_rgb, proprio, deterministic=True)
            progress_target = visual_progress_target(observation)
        target_image = base_rgb.float().div(255.0)
        identity_image = agent.canonicalize(base_rgb).div(255.0)
        canonical_images = [agent.canonicalize(rgb).div(255.0) for rgb in domain_rgbs]
        canonical_latents = [agent.base.encode(image.mul(255.0)) for image in canonical_images]
        canonical_actions = [
            action_from_latent(agent, latent, proprio) for latent in canonical_latents
        ]

        image_loss = torch.stack([
            F.l1_loss(image, target_image) for image in canonical_images
        ]).mean()
        edge_loss = torch.stack([
            image_gradient_loss(image, target_image) for image in canonical_images
        ]).mean()
        identity_loss = F.l1_loss(identity_image, target_image)
        action_loss = torch.stack([
            F.mse_loss(action, target_action) for action in canonical_actions
        ]).mean()
        feature_loss = torch.stack([
            F.smooth_l1_loss(latent, target_latent, beta=0.1)
            for latent in canonical_latents
        ]).mean()
        progress_loss = torch.stack([
            F.binary_cross_entropy_with_logits(
                agent.goal_progress_predictor(latent), progress_target,
            ) for latent in canonical_latents
        ]).mean()
        base_logits = agent.router(base_rgb)
        domain_logits = [agent.router(rgb) for rgb in domain_rgbs]
        router_loss = F.binary_cross_entropy_with_logits(
            base_logits, torch.zeros_like(base_logits),
        ) + torch.stack([
            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            for logits in domain_logits
        ]).mean()
        base_route_accuracy = (base_logits < 0).float().mean()
        domain_route_accuracy = torch.stack([
            (logits >= 0).float().mean() for logits in domain_logits
        ]).mean()
        loss = (
            float(task["image_weight"]) * image_loss
            + float(task["edge_weight"]) * edge_loss
            + float(task["identity_weight"]) * identity_loss
            + float(task["action_weight"]) * action_loss
            + float(task["feature_weight"]) * feature_loss
            + float(task["progress_weight"]) * progress_loss
            + float(task["router_weight"]) * router_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        with torch.no_grad():
            student_probability = float(task["student_rollout_max"]) * update / max(updates - 1, 1)
            student_mask = torch.rand((count, 1), device=device) < student_probability
            selected = canonical_actions[update % len(canonical_actions)].detach()
            executed = torch.where(student_mask, selected, target_action)
            observation, _, _, _, _ = env.step(executed)
        history.append({
            "loss": float(loss.detach()),
            "image_loss": float(image_loss.detach()),
            "edge_loss": float(edge_loss.detach()),
            "identity_loss": float(identity_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "feature_loss": float(feature_loss.detach()),
            "progress_loss": float(progress_loss.detach()),
            "router_loss": float(router_loss.detach()),
            "base_route_accuracy": float(base_route_accuracy.detach()),
            "domain_route_accuracy": float(domain_route_accuracy.detach()),
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
    best_metrics = {
        f"mean_last_100_{key}": float(np.mean([item[key] for item in recent]))
        for key in history[-1]
    }
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "canonical_agent": hashlib.sha256(
            Path(__file__).with_name("v33_canonical_view_agent.py").read_bytes()
        ).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(Path(canonical_environment.__file__).read_bytes()).hexdigest(),
        "multicamera_environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1,
        "training_protocol": "paired_canonical_view_v19_control",
        "observation_contract": observation_contract(task),
        "source_sha256": source_hashes,
        "task": task,
        "agent": agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": updates,
        "global_step": expected_transitions,
        "best_score": -best_metrics["mean_last_100_loss"],
        "best_metrics": best_metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    completion = {
        "schema_version": 1,
        "training_protocol": "paired_canonical_view_v19_control",
        "global_step": expected_transitions,
        "dagger_updates": updates,
        "dagger_environment_transitions": expected_transitions,
        "student_transitions": expected_transitions,
        "simulator_transitions": expected_transitions,
        "paired_view_training_samples": expected_transitions * len(domain_rgbs),
        "ppo_environment_steps": 0,
        "camera_keys": camera_keys,
        "sensor_augmentations": augmentations,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False,
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
