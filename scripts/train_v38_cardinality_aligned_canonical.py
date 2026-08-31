#!/usr/bin/env python3
"""Train V38 with factorized sensor cases and cardinality-aligned render pairs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v35_visual_recovery_unseen_ood import install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_visual_recovery_dual_teacher_ppo import (
    atomic_save, env_kwargs, extract_observation, file_sha256,
    observation_contract, privileged_aux_dim, select_task,
)
from train_v36_continuous_canonical import action_from_agent
from train_v37_dense_canonical import factorized_corruption, vector_env
from v37_dense_canonical_agent import DenseCanonicalV19Agent


def cumulative_source_interactions(completion: dict) -> int:
    for key in (
        "total_simulator_transitions", "protocol_environment_transitions_consumed",
        "total_environment_transitions", "simulator_transitions", "global_step",
    ):
        if key in completion:
            return int(completion[key])
    raise ValueError("source completion lacks interaction accounting")


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
        print(json.dumps({"task_count": task_count, **task}, indent=2)); return
    if not torch.cuda.is_available():
        raise RuntimeError("V38 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    install_extensions()
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V38 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    primary_env = vector_env(task["training_env_id"], count, env_kwargs(task))
    reference_env = vector_env(task["reference_env_id"], count, env_kwargs(task))
    profile_envs = {}
    for profile in task["paired_environment_profiles"]:
        kwargs = env_kwargs(task); kwargs["visual_domain_profile"] = profile
        profile_envs[profile] = vector_env("LearnedRecovery-v3-OOD", count, kwargs)
    primary_observation, _ = primary_env.reset(seed=seed)
    reference_observation, _ = reference_env.reset(seed=seed)
    profile_observations = {profile: env.reset(seed=seed)[0] for profile, env in profile_envs.items()}
    for shifted in profile_observations.values():
        assert_synchronized(reference_observation, shifted, task)
    _, proprio, critic = extract_observation(
        primary_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(primary_env.single_action_space.shape))
    agent = DenseCanonicalV19Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("training_protocol") != "continuous_similarity_photometric_repair_v19":
        raise ValueError("V38 source must be the V36 smoke checkpoint")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V38 source observation contract mismatch")
    agent.initialize_from_v36(source["agent"])
    optimizer = torch.optim.AdamW(
        list(agent.global_canonicalizer.parameters()) + list(agent.dense_residual.parameters()),
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]), eps=1e-5,
    )
    updates = int(task["canonical_updates"])
    per_update_simulator = count * (2 + len(profile_envs))
    transitions = updates * per_update_simulator
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V38 simulator-transition budget mismatch")
    history = []
    paired_reset_count = 0
    agent.train(); agent.base.eval()
    for update in range(updates):
        primary_rgb, primary_proprio, _ = extract_observation(
            primary_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        reference_rgb, reference_proprio, _ = extract_observation(
            reference_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = primary_observation["sensor_data"]
        synthetic, parameters, gain, bias = factorized_corruption(primary_rgb, task)
        geometry_pairs = [
            (synthetic, primary_rgb, primary_proprio),
            (views["camera_left_5cm"]["rgb"], primary_rgb, primary_proprio),
            (views["camera_high_5cm"]["rgb"], primary_rgb, primary_proprio),
        ]
        rendered_pairs = [
            (profile_observations[profile]["sensor_data"]["base_camera"]["rgb"], reference_rgb, reference_proprio)
            for profile in task["paired_environment_profiles"]
        ]
        paired_input, paired_target, paired_proprio = (geometry_pairs + rendered_pairs)[
            update % (len(geometry_pairs) + len(rendered_pairs))
        ]
        estimate = agent.global_canonicalizer.estimate(synthetic)
        maximum_log_scale = max(abs(math.log(task["minimum_scale"])), abs(math.log(task["maximum_scale"])))
        geometry_scale = torch.tensor([
            task["max_translation_pixels"], task["max_translation_pixels"],
            math.radians(task["max_rotation_degrees"]), maximum_log_scale,
        ], device=device)
        geometry_loss = F.smooth_l1_loss(estimate[1] / geometry_scale, parameters / geometry_scale, beta=0.05)
        color_loss = F.smooth_l1_loss(estimate[2].log(), gain.log(), beta=0.05)
        color_loss += F.smooth_l1_loss(estimate[3], bias, beta=0.05)
        corrected_synthetic = agent.correct(synthetic)
        corrected_paired = agent.correct(paired_input)
        corrected_primary = agent.correct(primary_rgb)
        corrected_reference = agent.correct(reference_rgb)
        image_loss = 0.5 * (
            F.l1_loss(corrected_synthetic.div(255.0), primary_rgb.float().div(255.0))
            + F.l1_loss(corrected_paired.div(255.0), paired_target.float().div(255.0))
        )
        identity_loss = 0.5 * (
            F.l1_loss(corrected_primary.div(255.0), primary_rgb.float().div(255.0))
            + F.l1_loss(corrected_reference.div(255.0), reference_rgb.float().div(255.0))
        )
        with torch.no_grad():
            primary_action, primary_latent = action_from_agent(agent, primary_rgb, primary_proprio)
            paired_action, paired_latent = action_from_agent(agent, paired_target, paired_proprio)
        synthetic_action, synthetic_latent = action_from_agent(agent, corrected_synthetic, primary_proprio)
        corrected_action, corrected_latent = action_from_agent(agent, corrected_paired, paired_proprio)
        action_loss = 0.5 * (F.mse_loss(synthetic_action, primary_action) + F.mse_loss(corrected_action, paired_action))
        feature_loss = 0.5 * (
            F.smooth_l1_loss(synthetic_latent, primary_latent, beta=0.1)
            + F.smooth_l1_loss(corrected_latent, paired_latent, beta=0.1)
        )
        loss = (
            task["geometry_weight"] * geometry_loss + task["color_weight"] * color_loss
            + task["image_weight"] * image_loss + task["identity_weight"] * identity_loss
            + task["action_weight"] * action_loss + task["feature_weight"] * feature_loss
        )
        optimizer.zero_grad(); loss.backward()
        trainable = list(agent.global_canonicalizer.parameters()) + list(agent.dense_residual.parameters())
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        with torch.no_grad():
            primary_executed = agent.base.get_action(primary_rgb, primary_proprio, deterministic=True)
            reference_executed = agent.base.get_action(reference_rgb, reference_proprio, deterministic=True)
            primary_observation, _, _, _, _ = primary_env.step(primary_executed)
            reference_observation, _, reference_terminated, reference_truncated, _ = reference_env.step(reference_executed)
            paired_done = torch.logical_or(reference_terminated, reference_truncated)
            for profile, env in profile_envs.items():
                profile_observations[profile], _, terminated, truncated, _ = env.step(reference_executed)
                paired_done = torch.logical_or(paired_done, torch.logical_or(terminated, truncated))
            if bool(paired_done.any()):
                paired_reset_count += 1; paired_seed = seed + 20_000_000 + paired_reset_count
                reference_observation, _ = reference_env.reset(seed=paired_seed)
                profile_observations = {profile: env.reset(seed=paired_seed)[0] for profile, env in profile_envs.items()}
        if (update + 1) % int(task["synchronization_check_frequency"]) == 0:
            for shifted in profile_observations.values():
                assert_synchronized(reference_observation, shifted, task)
        history.append({
            "loss": float(loss.detach()), "geometry_loss": float(geometry_loss.detach()),
            "color_loss": float(color_loss.detach()), "image_loss": float(image_loss.detach()),
            "identity_loss": float(identity_loss.detach()), "action_loss": float(action_loss.detach()),
            "feature_loss": float(feature_loss.detach()),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "simulator_transitions": (update + 1) * per_update_simulator}
            payload.update({key: float(np.mean([item[key] for item in recent])) for key in history[-1]})
            with (run_dir / "metrics.jsonl").open("a") as handle: handle.write(json.dumps(payload) + "\n")
    primary_env.close(); reference_env.close()
    for env in profile_envs.values(): env.close()
    recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([item[key] for item in recent])) for key in history[-1]}
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dense_agent": hashlib.sha256(Path(__file__).with_name("v37_dense_canonical_agent.py").read_bytes()).hexdigest(),
        "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1, "training_protocol": "cardinality_aligned_dense_repair_v19",
        "observation_contract": observation_contract(task), "source_sha256": source_hashes,
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": updates, "global_step": transitions, "best_score": -metrics["mean_last_100_loss"],
        "best_metrics": metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    source_completion = json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text())
    initialization = cumulative_source_interactions(source_completion)
    completion = {
        "schema_version": 1, "training_protocol": "cardinality_aligned_dense_repair_v19",
        "global_step": transitions, "dense_canonical_updates": updates,
        "dense_canonical_training_transitions": transitions, "simulator_transitions": transitions,
        "ppo_environment_steps": 0, "initialization_simulator_transitions": initialization,
        "total_simulator_transitions": initialization + transitions,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False, "source_visual_checkpoint": str(source_path),
        "source_visual_checkpoint_sha256": file_sha256(source_path), **metrics, "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
