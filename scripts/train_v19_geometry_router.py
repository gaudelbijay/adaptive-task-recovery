#!/usr/bin/env python3
"""Train a V19-preserving learned RGB router and geometry-grounded adapter."""

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
    VisualAgent,
    atomic_save,
    env_kwargs,
    extract_observation,
    file_sha256,
    observation_contract,
    privileged_aux_dim,
    privileged_representation_target,
    select_task,
    visual_progress_target,
)
from v32_hybrid_domain_agent import HybridDomainAgent


def robust_forward(agent, rgb, proprio):
    latent = agent.robust.encode(rgb)
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1))), latent


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
        raise RuntimeError("V32 geometry-router training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    canonical_environment = importlib.import_module("atr.envs.learned_recovery_v3")
    camera_keys = list(task["camera_keys"])
    if camera_keys != ["base_camera", "camera_left_5cm", "camera_high_5cm"]:
        raise ValueError("V32 requires the frozen same-physics camera ordering")
    augmentations = list(task["sensor_augmentations"])
    if augmentations != [
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    ]:
        raise ValueError("V32 requires the observed sensor-domain ordering")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V32 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    env = gym.make(task["training_env_id"], num_envs=count, **env_kwargs(task))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, count, record_metrics=True)
    observation, _ = env.reset(seed=seed)
    if set(observation["sensor_data"]) != set(camera_keys):
        raise ValueError("V32 multicamera observation keys mismatch")
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    aux_dim = privileged_aux_dim(task)
    if aux_dim != 14:
        raise ValueError("V32 requires the 14-dimensional geometry target")
    agent = HybridDomainAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, aux_dim,
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    teacher = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, aux_dim,
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V32 source visual contract mismatch")
    agent.initialize_from_v19(source["agent"])
    teacher.load_state_dict(source["agent"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(config["learning_rate"]), eps=1e-5)

    updates = int(task["dagger_updates"])
    expected_transitions = updates * count
    if int(task["total_timesteps"]) != expected_transitions:
        raise ValueError("V32 declared transition budget is inconsistent")
    rollout_max = float(task["student_rollout_max"])
    history = []
    agent.train()
    agent.base.eval()
    teacher.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {key: observation["sensor_data"][key]["rgb"] for key in camera_keys}
        base_rgb = views["base_camera"]
        augmentation = augmentations[update % len(augmentations)]
        shifted_rgb = apply_sensor_shift(base_rgb, augmentation)
        domain_rgbs = [views[camera_keys[1]], views[camera_keys[2]], shifted_rgb]
        with torch.no_grad():
            target_action = teacher.get_action(base_rgb, proprio, deterministic=True)
            progress_target = visual_progress_target(observation)
            # The shared helper scales Cartesian coordinates by five for the
            # original auxiliary head.  Undo that scale here and retain the
            # two binary resolution labels, yielding a bounded target.  The
            # tanh prediction below prevents an uncalibrated inherited head
            # from dominating the adapter gradients.
            geometry_target = privileged_representation_target(observation).clone()
            geometry_target[:, :12].div_(5.0)

        base_action, base_latent = robust_forward(agent, base_rgb, proprio)
        domain_outputs = [robust_forward(agent, rgb, proprio) for rgb in domain_rgbs]
        domain_actions = [item[0] for item in domain_outputs]
        domain_latents = [item[1] for item in domain_outputs]
        source_action_loss = F.mse_loss(base_action, target_action)
        domain_action_loss = torch.stack([
            F.mse_loss(action, target_action) for action in domain_actions
        ]).mean()
        progress_loss = torch.stack([
            F.binary_cross_entropy_with_logits(
                agent.goal_progress_predictor(latent), progress_target,
            ) for latent in [base_latent, *domain_latents]
        ]).mean()
        geometry_loss = torch.stack([
            F.smooth_l1_loss(
                torch.tanh(agent.robust.privileged_predictor(latent)),
                geometry_target,
                beta=0.1,
            )
            for latent in [base_latent, *domain_latents]
        ]).mean()
        invariant_target = base_latent.detach()
        multiview_invariance_loss = torch.stack([
            F.smooth_l1_loss(latent, invariant_target, beta=0.1)
            for latent in domain_latents
        ]).mean()

        with torch.no_grad():
            base_features = agent.base.encode(base_rgb)
            domain_features = [agent.base.encode(rgb) for rgb in domain_rgbs]
        base_logits = agent.router(base_features)
        domain_logits = [agent.router(features) for features in domain_features]
        router_loss = F.binary_cross_entropy_with_logits(
            base_logits, torch.zeros_like(base_logits),
        ) + torch.stack([
            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            for logits in domain_logits
        ]).mean()
        router_accuracy = torch.stack([
            (base_logits < 0).float().mean(),
            *[(logits >= 0).float().mean() for logits in domain_logits],
        ]).mean()

        loss = (
            float(task["source_action_weight"]) * source_action_loss
            + float(task["domain_action_weight"]) * domain_action_loss
            + float(task["progress_weight"]) * progress_loss
            + float(task["geometry_weight"]) * geometry_loss
            + float(task["multiview_invariance_weight"]) * multiview_invariance_loss
            + float(task["router_weight"]) * router_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        with torch.no_grad():
            student_probability = rollout_max * update / max(updates - 1, 1)
            student_mask = torch.rand((count, 1), device=device) < student_probability
            selected_domain_action = domain_actions[update % len(domain_actions)].detach()
            executed = torch.where(student_mask, selected_domain_action, target_action)
            observation, _, _, _, _ = env.step(executed)
        history.append({
            "loss": float(loss.detach()),
            "source_action_loss": float(source_action_loss.detach()),
            "domain_action_loss": float(domain_action_loss.detach()),
            "progress_loss": float(progress_loss.detach()),
            "geometry_loss": float(geometry_loss.detach()),
            "multiview_invariance_loss": float(multiview_invariance_loss.detach()),
            "router_loss": float(router_loss.detach()),
            "router_accuracy": float(router_accuracy.detach()),
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
        "hybrid_agent": hashlib.sha256(
            Path(__file__).with_name("v32_hybrid_domain_agent.py").read_bytes()
        ).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(Path(canonical_environment.__file__).read_bytes()).hexdigest(),
        "multicamera_environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1,
        "training_protocol": "v19_preserving_geometry_routed_dagger",
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
        "training_protocol": "v19_preserving_geometry_routed_dagger",
        "global_step": expected_transitions,
        "dagger_updates": updates,
        "dagger_environment_transitions": expected_transitions,
        "student_transitions": expected_transitions,
        "simulator_transitions": expected_transitions,
        "ppo_environment_steps": 0,
        "camera_keys": camera_keys,
        "sensor_augmentations": augmentations,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "training_only_geometry_target": True,
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
