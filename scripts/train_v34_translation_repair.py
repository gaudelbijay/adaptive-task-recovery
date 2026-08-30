#!/usr/bin/env python3
"""Train a supervised RGB translation repair while freezing the V34 policy."""

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
)
from v35_translation_repair_agent import (
    TranslationRepairedV34Agent, synthesize_content_translation,
)


def action_from_agent(agent, rgb, proprio):
    latent = agent.encode(rgb)
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
        raise RuntimeError("V35 translation repair requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    seed = int(task["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V35 run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")

    count = int(task["num_envs"])
    env = gym.make(task["training_env_id"], num_envs=count, **env_kwargs(task))
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, count, record_metrics=True)
    observation, _ = env.reset(seed=seed)
    camera_keys = list(task["camera_keys"])
    if set(observation["sensor_data"]) != set(camera_keys):
        raise ValueError("V35 multicamera observation keys mismatch")
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = TranslationRepairedV34Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    source_path = Path(str(task["source_v34_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V35 source observation contract mismatch")
    if source.get("training_protocol") != "factorized_canonical_v19_control":
        raise ValueError("V35 requires an audited V34 source checkpoint")
    source_completion_path = source_path.parent / "TRAINING_COMPLETE.json"
    source_completion = json.loads(source_completion_path.read_text(encoding="utf-8"))
    if source_completion.get("training_protocol") != "factorized_canonical_v19_control":
        raise ValueError("V35 source completion provenance mismatch")
    agent.initialize_from_v34(source["agent"])
    optimizer = torch.optim.Adam(
        agent.translation.parameters(), lr=float(config["learning_rate"]), eps=1e-5,
    )
    offset_catalog = torch.tensor(
        task["translation_training_offsets"], device=device, dtype=torch.float32,
    )
    if offset_catalog.ndim != 2 or offset_catalog.shape[1] != 2:
        raise ValueError("translation_training_offsets must be Nx2")
    if not bool((offset_catalog == torch.tensor([4.0, 0.0], device=device)).all(dim=1).any()):
        raise ValueError("V35 must include the observed right-four training shift")

    updates = int(task["translation_updates"])
    transitions = updates * count
    if int(task["total_timesteps"]) != transitions:
        raise ValueError("V35 transition budget mismatch")
    history = []
    agent.train()
    agent.robust.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {key: observation["sensor_data"][key]["rgb"] for key in camera_keys}
        base_rgb = views["base_camera"]
        random_indices = torch.randint(offset_catalog.shape[0], (count,), device=device)
        random_offsets = offset_catalog[random_indices]
        exact_offsets = torch.tensor([4.0, 0.0], device=device).repeat(count, 1)
        exact_shifted = synthesize_content_translation(base_rgb, exact_offsets)
        random_shifted = synthesize_content_translation(base_rgb, random_offsets)
        negatives = [
            base_rgb, views["camera_left_5cm"], views["camera_high_5cm"],
            apply_sensor_shift(base_rgb, "brightness_70"),
            apply_sensor_shift(base_rgb, "warm_color_shift"),
        ]
        positive_inputs = [exact_shifted, random_shifted]
        positive_targets = [exact_offsets, random_offsets]
        positive_outputs = [agent.translation(rgb) for rgb in positive_inputs]
        negative_outputs = [agent.translation(rgb) for rgb in negatives]
        classification_loss = 0.5 * (
            torch.stack([
                F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
                for logit, _ in positive_outputs
            ]).mean()
            + torch.stack([
                F.binary_cross_entropy_with_logits(logit, torch.zeros_like(logit))
                for logit, _ in negative_outputs
            ]).mean()
        )
        regression_loss = torch.stack([
            F.smooth_l1_loss(offset / 8.0, target / 8.0, beta=0.1)
            for (_, offset), target in zip(positive_outputs, positive_targets)
        ]).mean()
        negative_offset_loss = torch.stack([
            (offset / 8.0).square().mean() for _, offset in negative_outputs
        ]).mean()
        corrected_images = [
            agent.correct_translation(rgb, hard_route=False)[0]
            for rgb in positive_inputs
        ]
        image_loss = torch.stack([
            F.l1_loss(corrected.div(255.0), base_rgb.float().div(255.0))
            for corrected in corrected_images
        ]).mean()
        with torch.no_grad():
            target_action, target_latent = action_from_agent(agent.robust, base_rgb, proprio)
        corrected_actions_latents = [
            action_from_agent(agent.robust, corrected, proprio)
            for corrected in corrected_images
        ]
        action_loss = torch.stack([
            F.mse_loss(action, target_action)
            for action, _ in corrected_actions_latents
        ]).mean()
        feature_loss = torch.stack([
            F.smooth_l1_loss(latent, target_latent, beta=0.1)
            for _, latent in corrected_actions_latents
        ]).mean()
        shift_accuracy = torch.stack([
            (logit >= 0).float().mean() for logit, _ in positive_outputs
        ]).mean()
        negative_accuracy = torch.stack([
            (logit < 0).float().mean() for logit, _ in negative_outputs
        ]).mean()
        exact_offset_error = (positive_outputs[0][1] - exact_offsets).abs().mean()
        loss = (
            float(task["classification_weight"]) * classification_loss
            + float(task["regression_weight"]) * regression_loss
            + float(task["negative_offset_weight"]) * negative_offset_loss
            + float(task["image_weight"]) * image_loss
            + float(task["action_weight"]) * action_loss
            + float(task["feature_weight"]) * feature_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.translation.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            executed, _ = action_from_agent(agent.robust, base_rgb, proprio)
            observation, _, _, _, _ = env.step(executed)
        history.append({
            "loss": float(loss.detach()),
            "classification_loss": float(classification_loss.detach()),
            "regression_loss": float(regression_loss.detach()),
            "negative_offset_loss": float(negative_offset_loss.detach()),
            "image_loss": float(image_loss.detach()),
            "action_loss": float(action_loss.detach()),
            "feature_loss": float(feature_loss.detach()),
            "shift_accuracy": float(shift_accuracy.detach()),
            "negative_accuracy": float(negative_accuracy.detach()),
            "exact_offset_error_pixels": float(exact_offset_error.detach()),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "training_transitions": (update + 1) * count}
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
        "translation_agent": hashlib.sha256(
            Path(__file__).with_name("v35_translation_repair_agent.py").read_bytes()
        ).hexdigest(),
        "v34_agent": hashlib.sha256(
            Path(__file__).with_name("v34_factorized_canonical_agent.py").read_bytes()
        ).hexdigest(),
        "multicamera_environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
        "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1, "training_protocol": "supervised_translation_repair_v34",
        "observation_contract": observation_contract(task), "source_sha256": source_hashes,
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": updates, "global_step": transitions,
        "best_score": -best_metrics["mean_last_100_loss"], "best_metrics": best_metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    completion = {
        "schema_version": 1, "training_protocol": "supervised_translation_repair_v34",
        "global_step": transitions, "translation_updates": updates,
        "translation_training_transitions": transitions,
        "simulator_transitions": transitions, "ppo_environment_steps": 0,
        "initialization_simulator_transitions": int(source_completion["simulator_transitions"]),
        "total_simulator_transitions": transitions + int(source_completion["simulator_transitions"]),
        "translation_training_offsets": task["translation_training_offsets"],
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False,
        "source_v34_checkpoint": str(source_path),
        "source_v34_checkpoint_sha256": file_sha256(source_path),
        **best_metrics, "source_sha256": source_hashes,
    }
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
