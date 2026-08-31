#!/usr/bin/env python3
"""Train continuous RGB canonicalization while preserving frozen V19 control."""

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

from train_visual_recovery_dual_teacher_ppo import (
    atomic_save, env_kwargs, extract_observation, file_sha256,
    observation_contract, privileged_aux_dim, select_task,
)
from v36_continuous_canonical_agent import (
    ContinuousCanonicalV19Agent, synthesize_corruption,
)


def action_from_agent(agent, rgb, proprio):
    latent = agent.base.encode(rgb)
    parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(parts, dim=1))), latent


def source_interactions(completion: dict) -> int:
    for key in (
        "protocol_environment_transitions_consumed", "total_environment_transitions",
        "simulator_transitions", "global_step",
    ):
        if key in completion:
            return int(completion[key])
    raise ValueError("V19 completion lacks interaction accounting")


def main() -> None:
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
        raise RuntimeError("V36 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V36 run: {run_dir}")
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
        raise ValueError("V36 multicamera keys mismatch")
    _, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = ContinuousCanonicalV19Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).to(device)
    agent.canonicalizer.max_translation = float(task["max_translation_pixels"])
    agent.canonicalizer.max_rotation = math.radians(float(task["max_rotation_degrees"]))
    agent.canonicalizer.max_log_scale = max(
        abs(math.log(float(task["minimum_scale"]))),
        abs(math.log(float(task["maximum_scale"]))),
    )
    agent.canonicalizer.route_threshold = float(task["route_threshold"])
    source_path = Path(str(task["source_visual_checkpoint"]).format(seed=seed))
    source = torch.load(source_path, map_location=device, weights_only=False)
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V36 source observation contract mismatch")
    agent.initialize_from_v19(source["agent"])
    source_completion = json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text())
    optimizer = torch.optim.AdamW(
        agent.canonicalizer.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]), eps=1e-5,
    )

    updates = int(task["canonical_updates"])
    transitions = updates * count
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V36 transition budget mismatch")
    history = []
    agent.train(); agent.base.eval()
    for update in range(updates):
        _, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        views = {key: observation["sensor_data"][key]["rgb"] for key in camera_keys}
        base_rgb = views["base_camera"]
        parameters = torch.empty((count, 4), device=device)
        parameters[:, 0:2].uniform_(-float(task["max_translation_pixels"]), float(task["max_translation_pixels"]))
        parameters[:, 2].uniform_(-math.radians(float(task["max_rotation_degrees"])), math.radians(float(task["max_rotation_degrees"])))
        parameters[:, 3].uniform_(math.log(float(task["minimum_scale"])), math.log(float(task["maximum_scale"])))
        gain = torch.empty((count, 3), device=device).uniform_(
            math.log(float(task["minimum_color_gain"])), math.log(float(task["maximum_color_gain"])),
        ).exp()
        bias = torch.empty((count, 3), device=device).uniform_(
            -float(task["maximum_color_bias"]), float(task["maximum_color_bias"]),
        )
        corrupted = synthesize_corruption(base_rgb, parameters, gain, bias)
        positive_inputs = [corrupted, views["camera_left_5cm"], views["camera_high_5cm"]]
        positive_estimates = [agent.canonicalizer.estimate(rgb) for rgb in positive_inputs]
        clean_estimate = agent.canonicalizer.estimate(base_rgb)
        route_loss = 0.5 * (
            torch.stack([
                F.binary_cross_entropy_with_logits(item[0], torch.ones_like(item[0]))
                for item in positive_estimates
            ]).mean()
            + float(task["clean_route_weight"]) * F.binary_cross_entropy_with_logits(
                clean_estimate[0], torch.zeros_like(clean_estimate[0]),
            )
        )
        predicted = positive_estimates[0]
        maximum_log_scale = max(
            abs(math.log(float(task["minimum_scale"]))),
            abs(math.log(float(task["maximum_scale"]))),
        )
        geometry_scale = torch.tensor([
            float(task["max_translation_pixels"]), float(task["max_translation_pixels"]),
            math.radians(float(task["max_rotation_degrees"])), maximum_log_scale,
        ], device=device)
        geometry_loss = F.smooth_l1_loss(
            predicted[1] / geometry_scale, parameters / geometry_scale, beta=0.05,
        )
        color_loss = F.smooth_l1_loss(predicted[2].log(), gain.log(), beta=0.05)
        color_loss = color_loss + F.smooth_l1_loss(predicted[3], bias, beta=0.05)
        corrected = [agent.canonicalizer.correct(rgb, hard_route=False)[0] for rgb in positive_inputs]
        clean_corrected = agent.canonicalizer.correct(base_rgb, hard_route=False)[0]
        target_image = base_rgb.float().div(255.0)
        image_loss = torch.stack([
            F.l1_loss(item.div(255.0), target_image) for item in corrected
        ]).mean()
        identity_loss = F.l1_loss(clean_corrected.div(255.0), target_image)
        with torch.no_grad():
            target_action, target_latent = action_from_agent(agent, base_rgb, proprio)
        action_latents = [action_from_agent(agent, item, proprio) for item in corrected]
        action_loss = torch.stack([F.mse_loss(a, target_action) for a, _ in action_latents]).mean()
        feature_loss = torch.stack([
            F.smooth_l1_loss(z, target_latent, beta=0.1) for _, z in action_latents
        ]).mean()
        clean_false_route = (torch.sigmoid(clean_estimate[0]) >= float(task["route_threshold"])).float().mean()
        positive_route = torch.stack([
            (torch.sigmoid(item[0]) >= float(task["route_threshold"])).float().mean()
            for item in positive_estimates
        ]).mean()
        loss = (
            float(task["route_weight"]) * route_loss
            + float(task["geometry_weight"]) * geometry_loss
            + float(task["color_weight"]) * color_loss
            + float(task["image_weight"]) * image_loss
            + float(task["identity_weight"]) * identity_loss
            + float(task["action_weight"]) * action_loss
            + float(task["feature_weight"]) * feature_loss
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.canonicalizer.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            executed = agent.base.get_action(base_rgb, proprio, deterministic=True)
            observation, _, _, _, _ = env.step(executed)
        history.append({
            "loss": float(loss.detach()), "route_loss": float(route_loss.detach()),
            "geometry_loss": float(geometry_loss.detach()), "color_loss": float(color_loss.detach()),
            "image_loss": float(image_loss.detach()), "identity_loss": float(identity_loss.detach()),
            "action_loss": float(action_loss.detach()), "feature_loss": float(feature_loss.detach()),
            "clean_false_route": float(clean_false_route.detach()),
            "positive_route": float(positive_route.detach()),
        })
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "training_transitions": (update + 1) * count}
            payload.update({key: float(np.mean([x[key] for x in recent])) for key in history[-1]})
            with (run_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
    env.close()
    recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([x[key] for x in recent])) for key in history[-1]}
    source_hashes = {
        "trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "continuous_agent": hashlib.sha256(Path(__file__).with_name("v36_continuous_canonical_agent.py").read_bytes()).hexdigest(),
        "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest(),
    }
    checkpoint = {
        "schema_version": 1, "training_protocol": "continuous_similarity_photometric_repair_v19",
        "observation_contract": observation_contract(task), "source_sha256": source_hashes,
        "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
        "iteration": updates, "global_step": transitions, "best_score": -metrics["mean_last_100_loss"],
        "best_metrics": metrics,
    }
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    initialization = source_interactions(source_completion)
    completion = {
        "schema_version": 1, "training_protocol": "continuous_similarity_photometric_repair_v19",
        "global_step": transitions, "canonical_updates": updates,
        "canonical_training_transitions": transitions, "simulator_transitions": transitions,
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
