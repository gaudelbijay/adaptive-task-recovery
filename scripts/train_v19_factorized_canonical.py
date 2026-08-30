#!/usr/bin/env python3
"""Train V34 dense-warp/photometric canonicalization in front of frozen V19."""

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
    atomic_save, env_kwargs, extract_observation, file_sha256,
    observation_contract, privileged_aux_dim, select_task,
    visual_progress_target,
)
from v34_factorized_canonical_agent import FactorizedCanonicalV19Agent


def action_from_latent(agent, latent, proprio):
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1)))


def make_vector_env(task, count, env_id, profile=None):
    kwargs = env_kwargs(task)
    if profile is not None:
        kwargs["visual_domain_profile"] = profile
    env = gym.make(env_id, num_envs=count, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, count, record_metrics=True)


def image_gradient_loss(prediction, target, weight):
    pred_x = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_x = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_y = prediction[:, 1:, :, :] - prediction[:, :-1, :, :]
    target_y = target[:, 1:, :, :] - target[:, :-1, :, :]
    weight_x = weight[:, :, 1:, :]
    weight_y = weight[:, 1:, :, :]
    return (
        ((pred_x - target_x).abs() * weight_x).sum() / (weight_x.sum() * 3.0)
        + ((pred_y - target_y).abs() * weight_y).sum() / (weight_y.sum() * 3.0)
    )


def foreground_image_loss(prediction, target, source, foreground_weight):
    changed = (source.float().div(255.0) - target).abs().mean(dim=-1, keepdim=True)
    weight = 1.0 + float(foreground_weight) * (changed >= 0.02).float()
    loss = ((prediction - target).abs() * weight).sum() / (weight.sum() * 3.0)
    return loss, weight


def flow_regularization(flow):
    smooth = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    smooth = smooth + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    return smooth, flow.abs().mean()


def assert_synchronized(reference, shifted, task):
    _, ref_proprio, ref_critic = extract_observation(
        reference, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    _, shifted_proprio, shifted_critic = extract_observation(
        shifted, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    if not torch.allclose(ref_proprio, shifted_proprio, atol=2e-5, rtol=1e-5):
        difference = float((ref_proprio - shifted_proprio).abs().max())
        raise RuntimeError(
            f"paired lighting environment proprioception diverged: max_abs={difference}"
        )
    if not torch.allclose(ref_critic, shifted_critic, atol=2e-5, rtol=1e-5):
        raise RuntimeError("paired lighting environment critic state diverged")
    if not torch.equal(visual_progress_target(reference), visual_progress_target(shifted)):
        raise RuntimeError("paired lighting environment task progress diverged")


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
        raise RuntimeError("V34 factorized canonical training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    ood_environment = importlib.import_module("atr.envs.learned_recovery_v3_ood")
    canonical_environment = importlib.import_module("atr.envs.learned_recovery_v3")
    camera_keys = list(task["camera_keys"])
    augmentations = list(task["sensor_augmentations"])
    lighting_profiles = list(task["paired_lighting_profiles"])
    if camera_keys != ["base_camera", "camera_left_5cm", "camera_high_5cm"]:
        raise ValueError("V34 requires the frozen same-physics camera ordering")
    if augmentations != ["pixel_shift_right_4", "brightness_70", "warm_color_shift"]:
        raise ValueError("V34 requires the complete observed sensor domains")
    if lighting_profiles != ["lighting_dim", "lighting_warm"]:
        raise ValueError("V34 requires the complete observed renderer-light domains")

    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V34 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    primary_env = make_vector_env(task, count, task["training_env_id"])
    reference_env = make_vector_env(task, count, task["reference_env_id"])
    lighting_envs = {
        profile: make_vector_env(task, count, task["lighting_env_id"], profile)
        for profile in lighting_profiles
    }
    observation, _ = primary_env.reset(seed=seed)
    reference_observation, _ = reference_env.reset(seed=seed)
    lighting_observations = {
        profile: env.reset(seed=seed)[0] for profile, env in lighting_envs.items()
    }
    if set(observation["sensor_data"]) != set(camera_keys):
        raise ValueError("V34 multicamera observation keys mismatch")
    for shifted in lighting_observations.values():
        assert_synchronized(reference_observation, shifted, task)
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(primary_env.single_action_space.shape))
    agent = FactorizedCanonicalV19Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V34 source visual contract mismatch")
    agent.initialize_from_v19(source["agent"])
    trainable = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=float(config["learning_rate"]), eps=1e-5)

    updates = int(task["dagger_updates"])
    primary_transitions = updates * count
    simulator_transitions = primary_transitions * (2 + len(lighting_envs))
    if int(task["total_timesteps"]) != primary_transitions:
        raise ValueError("V34 declared primary transition budget is inconsistent")
    history = []
    paired_reset_count = 0
    agent.train()
    agent.base.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {key: observation["sensor_data"][key]["rgb"] for key in camera_keys}
        base_rgb = views["base_camera"]
        reference_rgb, reference_proprio, _ = extract_observation(
            reference_observation, task["asymmetric_critic"],
            task.get("actor_tcp_pose", False), task.get("actor_goal_progress", False),
        )
        geometry_rgbs = [views[camera_keys[1]], views[camera_keys[2]]]
        geometry_rgbs.extend(apply_sensor_shift(base_rgb, mode) for mode in augmentations)
        lighting_rgbs = [
            lighting_observations[profile]["sensor_data"]["base_camera"]["rgb"]
            for profile in lighting_profiles
        ]
        domain_rgbs = [*geometry_rgbs, *lighting_rgbs]
        with torch.no_grad():
            target_latent = agent.base.encode(base_rgb)
            target_action = agent.base.get_action(base_rgb, proprio, deterministic=True)
            progress_target = visual_progress_target(observation)
            reference_latent = agent.base.encode(reference_rgb)
            reference_action = agent.base.get_action(
                reference_rgb, reference_proprio, deterministic=True,
            )
            reference_progress = visual_progress_target(reference_observation)
        target_image = base_rgb.float().div(255.0)
        reference_image = reference_rgb.float().div(255.0)
        identity_primary, identity_primary_flow = agent.canonicalize(
            base_rgb, return_flow=True,
        )
        identity_reference, identity_reference_flow = agent.canonicalize(
            reference_rgb, return_flow=True,
        )
        identity_primary = identity_primary.div(255.0)
        identity_reference = identity_reference.div(255.0)
        canonical_pairs = [agent.canonicalize(rgb, return_flow=True) for rgb in domain_rgbs]
        canonical_images = [image.div(255.0) for image, _ in canonical_pairs]
        flows = [flow for _, flow in canonical_pairs]
        canonical_latents = [agent.base.encode(image.mul(255.0)) for image in canonical_images]
        target_images = [target_image] * len(geometry_rgbs) + [reference_image] * len(lighting_rgbs)
        target_latents = [target_latent] * len(geometry_rgbs) + [reference_latent] * len(lighting_rgbs)
        target_actions = [target_action] * len(geometry_rgbs) + [reference_action] * len(lighting_rgbs)
        target_progresses = [progress_target] * len(geometry_rgbs) + [reference_progress] * len(lighting_rgbs)
        action_proprios = [proprio] * len(geometry_rgbs) + [reference_proprio] * len(lighting_rgbs)
        canonical_actions = [
            action_from_latent(agent, latent, action_proprio)
            for latent, action_proprio in zip(canonical_latents, action_proprios)
        ]

        image_items = []
        edge_items = []
        for image, target, source_rgb in zip(canonical_images, target_images, domain_rgbs):
            image_item, weight = foreground_image_loss(
                image, target, source_rgb, task["foreground_weight"],
            )
            image_items.append(image_item)
            edge_items.append(image_gradient_loss(image, target, weight))
        image_loss = torch.stack(image_items).mean()
        edge_loss = torch.stack(edge_items).mean()
        identity_loss = 0.5 * (
            F.l1_loss(identity_primary, target_image)
            + F.l1_loss(identity_reference, reference_image)
        )
        action_loss = torch.stack([
            F.mse_loss(action, target)
            for action, target in zip(canonical_actions, target_actions)
        ]).mean()
        feature_loss = torch.stack([
            F.smooth_l1_loss(latent, target, beta=0.1)
            for latent, target in zip(canonical_latents, target_latents)
        ]).mean()
        progress_loss = torch.stack([
            F.binary_cross_entropy_with_logits(
                agent.goal_progress_predictor(latent), target,
            ) for latent, target in zip(canonical_latents, target_progresses)
        ]).mean()
        smooth_items, magnitude_items = zip(*(flow_regularization(flow) for flow in flows))
        flow_smoothness = torch.stack(smooth_items).mean()
        flow_magnitude = torch.stack(magnitude_items).mean()
        identity_flow_loss = 0.5 * (
            identity_primary_flow.abs().mean() + identity_reference_flow.abs().mean()
        )
        base_router_logits = [agent.router(base_rgb), agent.router(reference_rgb)]
        domain_router_logits = [agent.router(rgb) for rgb in domain_rgbs]
        base_router_loss = torch.stack([
            F.cross_entropy(logits, torch.zeros(count, device=device, dtype=torch.long))
            for logits in base_router_logits
        ]).mean()
        domain_router_losses = [
            F.cross_entropy(logits, torch.full(
                (count,), index, device=device, dtype=torch.long,
            )) for index, logits in enumerate(domain_router_logits, start=1)
        ]
        router_loss = torch.stack([base_router_loss, *domain_router_losses]).mean()
        base_route_accuracy = torch.stack([
            (logits.argmax(dim=1) == 0).float().mean() for logits in base_router_logits
        ]).mean()
        domain_route_accuracy = torch.stack([
            (logits.argmax(dim=1) == index).float().mean()
            for index, logits in enumerate(domain_router_logits, start=1)
        ]).mean()
        loss = (
            float(task["image_weight"]) * image_loss
            + float(task["edge_weight"]) * edge_loss
            + float(task["identity_weight"]) * identity_loss
            + float(task["action_weight"]) * action_loss
            + float(task["feature_weight"]) * feature_loss
            + float(task["progress_weight"]) * progress_loss
            + float(task["router_weight"]) * router_loss
            + float(task["flow_smoothness_weight"]) * flow_smoothness
            + float(task["flow_magnitude_weight"]) * flow_magnitude
            + float(task["identity_flow_weight"]) * identity_flow_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        with torch.no_grad():
            student_probability = float(task["student_rollout_max"]) * update / max(updates - 1, 1)
            student_mask = torch.rand((count, 1), device=device) < student_probability
            primary_selected = canonical_actions[update % len(geometry_rgbs)].detach()
            primary_executed = torch.where(student_mask, primary_selected, target_action)
            lighting_selected = canonical_actions[
                len(geometry_rgbs) + (update % len(lighting_rgbs))
            ].detach()
            lighting_executed = torch.where(
                student_mask, lighting_selected, reference_action,
            )
            observation, _, _, _, _ = primary_env.step(primary_executed)
            (
                reference_observation, _, reference_terminated,
                reference_truncated, _,
            ) = reference_env.step(lighting_executed)
            paired_done = torch.logical_or(reference_terminated, reference_truncated)
            for profile, env in lighting_envs.items():
                (
                    lighting_observations[profile], _, lighting_terminated,
                    lighting_truncated, _,
                ) = env.step(lighting_executed)
                paired_done = torch.logical_or(
                    paired_done,
                    torch.logical_or(lighting_terminated, lighting_truncated),
                )
            if bool(paired_done.any()):
                paired_reset_count += 1
                paired_seed = seed + 10_000_000 + paired_reset_count
                reference_observation, _ = reference_env.reset(seed=paired_seed)
                lighting_observations = {
                    profile: env.reset(seed=paired_seed)[0]
                    for profile, env in lighting_envs.items()
                }
                for shifted in lighting_observations.values():
                    assert_synchronized(reference_observation, shifted, task)
        if (update + 1) % int(task["synchronization_check_frequency"]) == 0:
            for shifted in lighting_observations.values():
                assert_synchronized(reference_observation, shifted, task)
        history.append({
            "loss": float(loss.detach()), "image_loss": float(image_loss.detach()),
            "edge_loss": float(edge_loss.detach()), "identity_loss": float(identity_loss.detach()),
            "action_loss": float(action_loss.detach()), "feature_loss": float(feature_loss.detach()),
            "progress_loss": float(progress_loss.detach()), "router_loss": float(router_loss.detach()),
            "flow_smoothness": float(flow_smoothness.detach()),
            "flow_magnitude": float(flow_magnitude.detach()),
            "identity_flow": float(identity_flow_loss.detach()),
            "base_route_accuracy": float(base_route_accuracy.detach()),
            "domain_route_accuracy": float(domain_route_accuracy.detach()),
            "student_rollout_fraction": float(student_mask.float().mean()),
            "paired_reset_count": float(paired_reset_count),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "primary_transitions": (update + 1) * count}
            for key in history[-1]:
                payload[key] = float(np.mean([item[key] for item in recent]))
            with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

    primary_env.close()
    reference_env.close()
    for env in lighting_envs.values():
        env.close()
    recent = history[-100:]
    best_metrics = {
        f"mean_last_100_{key}": float(np.mean([item[key] for item in recent]))
        for key in history[-1]
    }
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "factorized_agent": hashlib.sha256(
            Path(__file__).with_name("v34_factorized_canonical_agent.py").read_bytes()
        ).hexdigest(),
        "base_visual_trainer": hashlib.sha256(
            Path(__file__).with_name("train_visual_recovery_dual_teacher_ppo.py").read_bytes()
        ).hexdigest(),
        "environment": hashlib.sha256(Path(canonical_environment.__file__).read_bytes()).hexdigest(),
        "multicamera_environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
        "ood_environment": hashlib.sha256(Path(ood_environment.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1, "training_protocol": "factorized_canonical_v19_control",
        "observation_contract": observation_contract(task), "source_sha256": source_hashes,
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": updates, "global_step": primary_transitions,
        "best_score": -best_metrics["mean_last_100_loss"], "best_metrics": best_metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    completion = {
        "schema_version": 1, "training_protocol": "factorized_canonical_v19_control",
        "global_step": primary_transitions, "dagger_updates": updates,
        "dagger_environment_transitions": primary_transitions,
        "student_transitions": primary_transitions,
        "simulator_transitions": simulator_transitions,
        "paired_view_training_samples": primary_transitions * len(domain_rgbs),
        "ppo_environment_steps": 0, "camera_keys": camera_keys,
        "sensor_augmentations": augmentations, "paired_lighting_profiles": lighting_profiles,
        "paired_explicit_reset_count": paired_reset_count,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False,
        "source_visual_checkpoint": str(source_path),
        "source_visual_checkpoint_sha256": file_sha256(source_path),
        **best_metrics, "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
