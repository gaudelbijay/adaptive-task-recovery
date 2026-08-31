#!/usr/bin/env python3
"""Fine-tune only V38's dense residual with back-key oversampling."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v35_visual_recovery_unseen_ood import install_extensions as install_v35_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_visual_recovery_dual_teacher_ppo import (
    atomic_save, env_kwargs, extract_observation, file_sha256,
    observation_contract, privileged_aux_dim, select_task,
)
from train_v36_continuous_canonical import action_from_agent
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2)); return
    if not torch.cuda.is_available():
        raise RuntimeError("V39 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    extension = task.get("environment_extension", "v35")
    if extension == "v35":
        install_v35_extensions()
    elif extension == "v41":
        from evaluate_v41_visual_recovery_unseen_ood import install_extensions
        install_extensions()
    else:
        raise ValueError(f"unknown environment extension: {extension}")
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V39 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    count = int(task["num_envs"])
    reference_env = vector_env(task["reference_env_id"], count, env_kwargs(task))
    profile_envs = {}
    for profile in task["paired_environment_profiles"]:
        kwargs = env_kwargs(task); kwargs["visual_domain_profile"] = profile
        profile_envs[profile] = vector_env("LearnedRecovery-v3-OOD", count, kwargs)
    reference_observation, _ = reference_env.reset(seed=seed)
    profile_observations = {profile: env.reset(seed=seed)[0] for profile, env in profile_envs.items()}
    for shifted in profile_observations.values(): assert_synchronized(reference_observation, shifted, task)
    reference_rgb, reference_proprio, critic = extract_observation(
        reference_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(reference_env.single_action_space.shape))
    agent = MagnitudeGatedDenseV19Agent(
        task["image_size"], reference_proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    agent.magnitude_threshold = float(task["magnitude_threshold"])
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    source_protocol = task.get(
        "source_training_protocol", "cardinality_aligned_dense_repair_v19",
    )
    if source.get("training_protocol") != source_protocol:
        raise ValueError("dense fine-tune source protocol mismatch")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V39 source observation contract mismatch")
    agent.load_state_dict(source["agent"], strict=True)
    for parameter in agent.parameters(): parameter.requires_grad_(False)
    for parameter in agent.dense_residual.parameters(): parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        agent.dense_residual.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), eps=1e-5,
    )
    updates = int(task["fine_tune_updates"])
    transitions = updates * count * (1 + len(profile_envs))
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V39 simulator-transition budget mismatch")
    sampling = list(task["profile_sampling_cycle"])
    history = []; reset_count = 0
    agent.train(); agent.base.eval(); agent.global_canonicalizer.eval()
    for update in range(updates):
        reference_rgb, reference_proprio, _ = extract_observation(
            reference_observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        profile = sampling[update % len(sampling)]
        shifted_rgb = profile_observations[profile]["sensor_data"]["base_camera"]["rgb"]
        corrected = agent.correct(shifted_rgb)
        clean_corrected = agent.correct(reference_rgb)
        image_loss = F.l1_loss(corrected.div(255.0), reference_rgb.float().div(255.0))
        identity_loss = F.l1_loss(clean_corrected.div(255.0), reference_rgb.float().div(255.0))
        with torch.no_grad(): target_action, target_latent = action_from_agent(agent, reference_rgb, reference_proprio)
        corrected_action, corrected_latent = action_from_agent(agent, corrected, reference_proprio)
        action_loss = F.mse_loss(corrected_action, target_action)
        feature_loss = F.smooth_l1_loss(corrected_latent, target_latent, beta=0.1)
        loss = (
            task["image_weight"] * image_loss + task["identity_weight"] * identity_loss
            + task["action_weight"] * action_loss + task["feature_weight"] * feature_loss
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.dense_residual.parameters(), 1.0); optimizer.step()
        with torch.no_grad():
            executed = agent.base.get_action(reference_rgb, reference_proprio, deterministic=True)
            reference_observation, _, terminated, truncated, _ = reference_env.step(executed)
            paired_done = torch.logical_or(terminated, truncated)
            for name, env in profile_envs.items():
                profile_observations[name], _, term, trunc, _ = env.step(executed)
                paired_done = torch.logical_or(paired_done, torch.logical_or(term, trunc))
            if bool(paired_done.any()):
                reset_count += 1; paired_seed = seed + 30_000_000 + reset_count
                reference_observation, _ = reference_env.reset(seed=paired_seed)
                profile_observations = {name: env.reset(seed=paired_seed)[0] for name, env in profile_envs.items()}
        if (update + 1) % int(task["synchronization_check_frequency"]) == 0:
            for shifted in profile_observations.values(): assert_synchronized(reference_observation, shifted, task)
        history.append({"loss":float(loss.detach()), "image_loss":float(image_loss.detach()),
                        "identity_loss":float(identity_loss.detach()), "action_loss":float(action_loss.detach()),
                        "feature_loss":float(feature_loss.detach())})
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update":update + 1, "simulator_transitions":(update + 1) * count * (1 + len(profile_envs))}
            payload.update({key:float(np.mean([item[key] for item in recent])) for key in history[-1]})
            with (run_dir / "metrics.jsonl").open("a") as handle: handle.write(json.dumps(payload) + "\n")
    reference_env.close()
    for env in profile_envs.values(): env.close()
    recent = history[-100:]
    metrics = {f"mean_last_100_{key}":float(np.mean([item[key] for item in recent])) for key in history[-1]}
    source_hashes = {"trainer":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "agent":hashlib.sha256(Path(__file__).with_name("v39_magnitude_gated_agent.py").read_bytes()).hexdigest(),
                     "environment":hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    training_protocol = task.get(
        "training_protocol", "backkey_targeted_dense_repair_v19",
    )
    checkpoint = {"schema_version":1, "training_protocol":training_protocol,
                  "observation_contract":observation_contract(task), "source_sha256":source_hashes,
                  "task":task, "agent":agent.state_dict(), "optimizer":optimizer.state_dict(),
                  "iteration":updates, "global_step":transitions, "best_score":-metrics["mean_last_100_loss"],
                  "best_metrics":metrics}
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    source_completion = json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text())
    initialization = cumulative_source_interactions(source_completion)
    completion = {"schema_version":1, "training_protocol":training_protocol,
                  "global_step":transitions, "dense_finetune_updates":updates,
                  "dense_finetune_transitions":transitions, "simulator_transitions":transitions,
                  "ppo_environment_steps":0, "initialization_simulator_transitions":initialization,
                  "total_simulator_transitions":initialization + transitions,
                  "deployment_actor_inputs":"rgb_qpos_qvel_tcp_instruction_learned_progress",
                  "evaluation_domain_label_available":False, "magnitude_threshold":task["magnitude_threshold"],
                  "source_visual_checkpoint":str(source_path), "source_visual_checkpoint_sha256":file_sha256(source_path),
                  **metrics, "source_sha256":source_hashes}
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__": main()
