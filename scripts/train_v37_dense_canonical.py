#!/usr/bin/env python3
"""Fine-tune V36 into an always-on dense canonicalizer using paired domains."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
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

from evaluate_v35_visual_recovery_unseen_ood import install_extensions
from train_visual_recovery_dual_teacher_ppo import (
    atomic_save, env_kwargs, extract_observation, file_sha256,
    observation_contract, privileged_aux_dim, select_task,
)
from train_v36_continuous_canonical import action_from_agent, source_interactions
from v36_continuous_canonical_agent import synthesize_corruption
from v37_dense_canonical_agent import DenseCanonicalV19Agent


def vector_env(env_id: str, count: int, kwargs: dict):
    env = gym.make(env_id, num_envs=count, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, count, record_metrics=True)


def factorized_corruption(rgb: torch.Tensor, task: dict):
    count = rgb.shape[0]
    parameters = torch.zeros((count, 4), device=rgb.device)
    gain = torch.ones((count, 3), device=rgb.device)
    bias = torch.zeros((count, 3), device=rgb.device)
    modes = torch.randint(0, 7, (count,), device=rgb.device)
    tx = torch.empty(count, device=rgb.device).uniform_(-task["max_translation_pixels"], task["max_translation_pixels"])
    ty = torch.empty(count, device=rgb.device).uniform_(-task["max_translation_pixels"], task["max_translation_pixels"])
    rotation = torch.empty(count, device=rgb.device).uniform_(
        -math.radians(task["max_rotation_degrees"]), math.radians(task["max_rotation_degrees"]),
    )
    scale = torch.empty(count, device=rgb.device).uniform_(
        math.log(task["minimum_scale"]), math.log(task["maximum_scale"]),
    )
    sampled_gain = torch.empty((count, 3), device=rgb.device).uniform_(
        math.log(task["minimum_color_gain"]), math.log(task["maximum_color_gain"]),
    ).exp()
    sampled_bias = torch.empty((count, 3), device=rgb.device).uniform_(
        -task["maximum_color_bias"], task["maximum_color_bias"],
    )
    parameters[:, 0] = torch.where((modes == 1) | (modes == 6), tx, parameters[:, 0])
    parameters[:, 1] = torch.where((modes == 2) | (modes == 6), ty, parameters[:, 1])
    parameters[:, 2] = torch.where((modes == 3) | (modes == 6), rotation, parameters[:, 2])
    parameters[:, 3] = torch.where((modes == 4) | (modes == 6), scale, parameters[:, 3])
    color_mask = ((modes == 5) | (modes == 6))[:, None]
    gain = torch.where(color_mask, sampled_gain, gain)
    bias = torch.where(color_mask, sampled_bias, bias)
    return synthesize_corruption(rgb, parameters, gain, bias), parameters, gain, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("V37 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    install_extensions()
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V37 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    base_env = vector_env(task["training_env_id"], count, env_kwargs(task))
    observations = []
    base_observation, _ = base_env.reset(seed=seed)
    profile_envs = []
    for profile in task["paired_environment_profiles"]:
        kwargs = env_kwargs(task); kwargs["visual_domain_profile"] = profile
        env = vector_env("LearnedRecovery-v3-OOD", count, kwargs)
        observation, _ = env.reset(seed=seed)
        profile_envs.append(env); observations.append(observation)
    _, proprio, critic = extract_observation(
        base_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(base_env.single_action_space.shape))
    agent = DenseCanonicalV19Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("training_protocol") != "continuous_similarity_photometric_repair_v19":
        raise ValueError("V37 source must be a V36 continuous-canonical checkpoint")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V37 source observation contract mismatch")
    agent.initialize_from_v36(source["agent"])
    agent.global_canonicalizer.max_translation = float(task["max_translation_pixels"])
    agent.global_canonicalizer.max_rotation = math.radians(float(task["max_rotation_degrees"]))
    agent.global_canonicalizer.max_log_scale = max(
        abs(math.log(float(task["minimum_scale"]))), abs(math.log(float(task["maximum_scale"]))),
    )
    optimizer = torch.optim.AdamW(
        list(agent.global_canonicalizer.parameters()) + list(agent.dense_residual.parameters()),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]), eps=1e-5,
    )
    updates = int(task["canonical_updates"])
    per_update_simulator = count * (1 + len(profile_envs))
    transitions = updates * per_update_simulator
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V37 simulator-transition budget mismatch")
    history = []
    agent.train(); agent.base.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            base_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = base_observation["sensor_data"]
        base_rgb = views["base_camera"]["rgb"]
        synthetic, parameters, gain, bias = factorized_corruption(base_rgb, task)
        paired_inputs = [views["camera_left_5cm"]["rgb"], views["camera_high_5cm"]["rgb"]]
        paired_inputs.extend(item["sensor_data"]["base_camera"]["rgb"] for item in observations)
        paired = paired_inputs[update % len(paired_inputs)]
        synthetic_estimate = agent.global_canonicalizer.estimate(synthetic)
        maximum_log_scale = max(abs(math.log(task["minimum_scale"])), abs(math.log(task["maximum_scale"])))
        geometry_scale = torch.tensor([
            task["max_translation_pixels"], task["max_translation_pixels"],
            math.radians(task["max_rotation_degrees"]), maximum_log_scale,
        ], device=device)
        geometry_loss = F.smooth_l1_loss(
            synthetic_estimate[1] / geometry_scale, parameters / geometry_scale, beta=0.05,
        )
        color_loss = F.smooth_l1_loss(synthetic_estimate[2].log(), gain.log(), beta=0.05)
        color_loss += F.smooth_l1_loss(synthetic_estimate[3], bias, beta=0.05)
        corrected_synthetic = agent.correct(synthetic)
        corrected_paired = agent.correct(paired)
        corrected_clean = agent.correct(base_rgb)
        target_image = base_rgb.float().div(255.0)
        image_loss = 0.5 * (
            F.l1_loss(corrected_synthetic.div(255.0), target_image)
            + F.l1_loss(corrected_paired.div(255.0), target_image)
        )
        identity_loss = F.l1_loss(corrected_clean.div(255.0), target_image)
        with torch.no_grad():
            target_action, target_latent = action_from_agent(agent, base_rgb, proprio)
        action_latents = [action_from_agent(agent, item, proprio) for item in (corrected_synthetic, corrected_paired)]
        action_loss = torch.stack([F.mse_loss(a, target_action) for a, _ in action_latents]).mean()
        feature_loss = torch.stack([F.smooth_l1_loss(z, target_latent, beta=0.1) for _, z in action_latents]).mean()
        loss = (
            task["geometry_weight"] * geometry_loss + task["color_weight"] * color_loss
            + task["image_weight"] * image_loss + task["identity_weight"] * identity_loss
            + task["action_weight"] * action_loss + task["feature_weight"] * feature_loss
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(agent.global_canonicalizer.parameters()) + list(agent.dense_residual.parameters()), 1.0,
        )
        optimizer.step()
        with torch.no_grad():
            executed = agent.base.get_action(base_rgb, proprio, deterministic=True)
            base_observation, _, _, _, _ = base_env.step(executed)
            observations = [env.step(executed)[0] for env in profile_envs]
        history.append({
            "loss": float(loss.detach()), "geometry_loss": float(geometry_loss.detach()),
            "color_loss": float(color_loss.detach()), "image_loss": float(image_loss.detach()),
            "identity_loss": float(identity_loss.detach()), "action_loss": float(action_loss.detach()),
            "feature_loss": float(feature_loss.detach()),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "simulator_transitions": (update + 1) * per_update_simulator}
            payload.update({key: float(np.mean([x[key] for x in recent])) for key in history[-1]})
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
    base_env.close()
    for env in profile_envs: env.close()
    recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([x[key] for x in recent])) for key in history[-1]}
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dense_agent": hashlib.sha256(Path(__file__).with_name("v37_dense_canonical_agent.py").read_bytes()).hexdigest(),
        "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1, "training_protocol": "dense_paired_domain_repair_v19",
        "observation_contract": observation_contract(task), "source_sha256": source_hashes,
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": updates, "global_step": transitions, "best_score": -metrics["mean_last_100_loss"],
        "best_metrics": metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    source_completion = json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text())
    initialization = source_interactions(source_completion)
    completion = {
        "schema_version": 1, "training_protocol": "dense_paired_domain_repair_v19",
        "global_step": transitions, "dense_canonical_updates": updates,
        "dense_canonical_training_transitions": transitions, "simulator_transitions": transitions,
        "ppo_environment_steps": 0, "initialization_simulator_transitions": initialization,
        "total_simulator_transitions": initialization + transitions,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False, "source_visual_checkpoint": str(source_path),
        "source_visual_checkpoint_sha256": file_sha256(source_path), **metrics,
        "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
