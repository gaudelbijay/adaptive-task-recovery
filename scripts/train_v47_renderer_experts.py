#!/usr/bin/env python3
"""Train a camera expert and RGB-only three-way renderer router."""

import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v41_visual_recovery_unseen_ood import apply_visual_perturbation, install_extensions
from train_v19_factorized_canonical import assert_synchronized
from train_v44_multiview_feature_adapter import action_from_latent
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from v47_renderer_expert_agent import RendererExpertV41Agent


GEOMETRIC_NEGATIVES = (
    "subpixel_shift_right_2_25", "rotation_counterclockwise_4deg",
    "scale_108", "combined_similarity_v1",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("V47 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    install_extensions()
    seed = int(task["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V47 run: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    n = int(task["num_envs"])
    reference = vector_env(task["reference_env_id"], n, env_kwargs(task))
    profiles = {}
    for name in task["paired_environment_profiles"]:
        kwargs = env_kwargs(task); kwargs["visual_domain_profile"] = name
        profiles[name] = vector_env("LearnedRecovery-v3-OOD", n, kwargs)
    reference_obs, _ = reference.reset(seed=seed)
    shifted = {name: env.reset(seed=seed)[0] for name, env in profiles.items()}
    for observation in shifted.values():
        assert_synchronized(reference_obs, observation, task)
    rgb, proprio, critic = extract_observation(
        reference_obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False), False
    )
    action_dim = int(np.prod(reference.single_action_space.shape))
    agent = RendererExpertV41Agent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    source_path = Path(task["source_visual_checkpoint"].format(seed=seed))
    source = torch.load(source_path, map_location="cuda", weights_only=False)
    if source.get("training_protocol") != "routed_multiview_feature_adapter_v19":
        raise ValueError("V47 requires the V45 feature-adapter source")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V47 observation contract mismatch")
    agent.initialize_from_v45(source["agent"])
    camera_parameters = list(agent.camera_encoder.parameters())
    router_parameters = list(agent.router.parameters())
    optimizer = torch.optim.AdamW([
        {"params": camera_parameters, "lr": config["camera_learning_rate"]},
        {"params": router_parameters, "lr": config["router_learning_rate"]},
    ], weight_decay=config["weight_decay"], eps=1e-5)
    updates = int(task["expert_updates"])
    transitions = updates * n * (1 + len(profiles))
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V47 transition budget mismatch")
    history = []; resets = 0; agent.train(); agent.v41.eval(); agent.lighting_encoder.eval()
    camera_name = task["camera_profile"]
    lighting_names = task["lighting_profiles"]
    for update in range(updates):
        rgb, proprio, _ = extract_observation(
            reference_obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False), False
        )
        camera_rgb = shifted[camera_name]["sensor_data"]["base_camera"]["rgb"]
        lighting_rgbs = [shifted[name]["sensor_data"]["base_camera"]["rgb"] for name in lighting_names]
        with torch.no_grad():
            target = agent.v41.encode(rgb)
            target_action = action_from_latent(agent, target, proprio)
        camera_latent = agent.camera_latent(camera_rgb)
        camera_action = action_from_latent(agent, camera_latent, proprio)
        feature_loss = F.smooth_l1_loss(camera_latent, target, beta=0.1)
        cosine_loss = (1 - F.cosine_similarity(camera_latent, target, dim=1)).mean()
        action_loss = F.mse_loss(camera_action, target_action)
        router_images = [rgb] + [apply_visual_perturbation(rgb, mode) for mode in GEOMETRIC_NEGATIVES]
        router_images += [camera_rgb] + lighting_rgbs
        labels = [0] * (1 + len(GEOMETRIC_NEGATIVES)) + [1] + [2] * len(lighting_rgbs)
        logits = agent.router_logits(torch.cat(router_images, dim=0))
        targets = torch.cat([
            torch.full((n,), label, device=logits.device, dtype=torch.long) for label in labels
        ])
        router_loss = F.cross_entropy(logits, targets)
        loss = (
            task["feature_weight"] * feature_loss
            + task["cosine_weight"] * cosine_loss
            + task["action_weight"] * action_loss
            + task["router_weight"] * router_loss
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(camera_parameters + router_parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == targets).float().mean()
            executed = agent.v41.base.get_action(rgb, proprio, deterministic=True)
            reference_obs, _, terminated, truncated, _ = reference.step(executed)
            done = torch.logical_or(terminated, truncated)
            for name, env in profiles.items():
                shifted[name], _, term, trunc, _ = env.step(executed)
                done = torch.logical_or(done, torch.logical_or(term, trunc))
            if bool(done.any()):
                resets += 1; paired_seed = seed + 50_000_000 + resets
                reference_obs, _ = reference.reset(seed=paired_seed)
                shifted = {name: env.reset(seed=paired_seed)[0] for name, env in profiles.items()}
        if (update + 1) % int(task["synchronization_check_frequency"]) == 0:
            for observation in shifted.values():
                assert_synchronized(reference_obs, observation, task)
        item = {"loss": float(loss.detach()), "feature_loss": float(feature_loss.detach()),
                "cosine_loss": float(cosine_loss.detach()), "action_loss": float(action_loss.detach()),
                "router_loss": float(router_loss.detach()), "router_accuracy": float(accuracy.detach())}
        history.append(item)
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "simulator_transitions": (update + 1) * n * (1 + len(profiles))}
            payload.update({key: float(np.mean([record[key] for record in recent])) for key in item})
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(payload) + "\n")
    reference.close(); [env.close() for env in profiles.values()]
    recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([record[key] for record in recent])) for key in history[-1]}
    hashes = {"trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "agent": hashlib.sha256(Path(__file__).with_name("v47_renderer_expert_agent.py").read_bytes()).hexdigest(),
              "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint = {"schema_version": 1, "training_protocol": "renderer_expert_adapter_v19",
                  "observation_contract": observation_contract(task), "source_sha256": hashes,
                  "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                  "iteration": updates, "global_step": transitions, "best_score": -metrics["mean_last_100_loss"],
                  "best_metrics": metrics}
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    initialization = cumulative_source_interactions(json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text()))
    completion = {"schema_version": 1, "training_protocol": "renderer_expert_adapter_v19",
                  "global_step": transitions, "feature_adapter_transitions": transitions,
                  "simulator_transitions": transitions, "ppo_environment_steps": 0,
                  "initialization_simulator_transitions": initialization,
                  "total_simulator_transitions": initialization + transitions,
                  "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
                  "evaluation_domain_label_available": False,
                  "source_visual_checkpoint": str(source_path),
                  "source_visual_checkpoint_sha256": file_sha256(source_path),
                  **metrics, "source_sha256": hashes}
    (run_dir / "TRAINING_COMPLETE.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
