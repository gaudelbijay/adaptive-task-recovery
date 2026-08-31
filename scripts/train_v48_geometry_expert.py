#!/usr/bin/env python3
"""Train only V48's geometry encoder and clean-vs-geometric RGB router."""

import argparse
import hashlib
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_v41_visual_recovery_unseen_ood import apply_visual_perturbation
from train_v44_multiview_feature_adapter import action_from_latent
from train_visual_recovery_dual_teacher_ppo import atomic_save, env_kwargs, extract_observation, file_sha256, observation_contract, privileged_aux_dim, select_task
from train_v37_dense_canonical import vector_env
from train_v38_cardinality_aligned_canonical import cumulative_source_interactions
from v48_geometry_expert_agent import HierarchicalGeometryExpertAgent


GEOMETRIC_MODES = (
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
        print(json.dumps({"task_count": task_count, **task}, indent=2)); return
    if not torch.cuda.is_available():
        raise RuntimeError("V48 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    seed = int(task["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V48 run: {run_dir}")
    run_dir.mkdir(parents=True); (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    n = int(task["num_envs"]); environment = vector_env(task["env_id"], n, env_kwargs(task))
    observation, _ = environment.reset(seed=seed)
    rgb, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False), False
    )
    action_dim = int(np.prod(environment.single_action_space.shape))
    agent = HierarchicalGeometryExpertAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    source_path = Path(task["source_visual_checkpoint"].format(seed=seed))
    source = torch.load(source_path, map_location="cuda", weights_only=False)
    if source.get("training_protocol") != "renderer_expert_adapter_v19":
        raise ValueError("V48 requires the V47 renderer-expert source")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V48 observation contract mismatch")
    agent.initialize_from_v47(source["agent"])
    encoder_parameters = list(agent.geometry_encoder.parameters())
    router_parameters = list(agent.geometry_router.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": config["geometry_learning_rate"]},
        {"params": router_parameters, "lr": config["router_learning_rate"]},
    ], weight_decay=config["weight_decay"], eps=1e-5)
    updates = int(task["geometry_updates"]); transitions = updates * n
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V48 transition budget mismatch")
    history = []; resets = 0; agent.train(); agent.v47.eval()
    for update in range(updates):
        rgb, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False), False
        )
        transformed = [apply_visual_perturbation(rgb, mode) for mode in GEOMETRIC_MODES]
        with torch.no_grad():
            target = agent.v47.v41.encode(rgb)
            target_action = action_from_latent(agent, target, proprio)
        latents = [agent.geometry_latent(image) for image in transformed]
        actions = [action_from_latent(agent, latent, proprio) for latent in latents]
        feature_loss = torch.stack([
            F.smooth_l1_loss(latent, target, beta=0.1) for latent in latents
        ]).mean()
        cosine_loss = torch.stack([
            (1 - F.cosine_similarity(latent, target, dim=1)).mean() for latent in latents
        ]).mean()
        action_loss = torch.stack([
            F.mse_loss(action, target_action) for action in actions
        ]).mean()
        clean_logits = agent.geometry_logits(rgb)
        shifted_logits = [agent.geometry_logits(image) for image in transformed]
        router_loss = F.binary_cross_entropy_with_logits(clean_logits, torch.zeros_like(clean_logits))
        router_loss = router_loss + torch.stack([
            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            for logits in shifted_logits
        ]).mean()
        loss = (task["feature_weight"] * feature_loss + task["cosine_weight"] * cosine_loss
                + task["action_weight"] * action_loss + task["router_weight"] * router_loss)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder_parameters + router_parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            clean_accuracy = (torch.sigmoid(clean_logits) < 0.5).float().mean()
            shifted_accuracy = torch.stack([
                (torch.sigmoid(logits) >= 0.5).float().mean() for logits in shifted_logits
            ]).mean()
            executed = agent.v47.v41.base.get_action(rgb, proprio, deterministic=True)
            observation, _, terminated, truncated, _ = environment.step(executed)
            if bool(torch.logical_or(terminated, truncated).any()):
                resets += 1; observation, _ = environment.reset(seed=seed + 60_000_000 + resets)
        item = {"loss": float(loss.detach()), "feature_loss": float(feature_loss.detach()),
                "cosine_loss": float(cosine_loss.detach()), "action_loss": float(action_loss.detach()),
                "router_loss": float(router_loss.detach()),
                "clean_router_accuracy": float(clean_accuracy.detach()),
                "shifted_router_accuracy": float(shifted_accuracy.detach())}
        history.append(item)
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "simulator_transitions": (update + 1) * n}
            payload.update({key: float(np.mean([record[key] for record in recent])) for key in item})
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(payload) + "\n")
    environment.close(); recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([record[key] for record in recent])) for key in history[-1]}
    hashes = {"trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "agent": hashlib.sha256(Path(__file__).with_name("v48_geometry_expert_agent.py").read_bytes()).hexdigest(),
              "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint = {"schema_version": 1, "training_protocol": "hierarchical_geometry_expert_v19",
                  "observation_contract": observation_contract(task), "source_sha256": hashes,
                  "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                  "iteration": updates, "global_step": transitions,
                  "best_score": -metrics["mean_last_100_loss"], "best_metrics": metrics}
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    initialization = cumulative_source_interactions(json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text()))
    completion = {"schema_version": 1, "training_protocol": "hierarchical_geometry_expert_v19",
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
