#!/usr/bin/env python3
"""On-policy corrective imitation for the frozen V48 geometry route."""

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
        raise RuntimeError("V49 training requires CUDA")
    registration = importlib.import_module(task["registration_module"])
    seed = int(task["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V49 run: {run_dir}")
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
    if source.get("training_protocol") != "hierarchical_geometry_expert_v19":
        raise ValueError("V49 requires the V48 hierarchical source")
    if source.get("observation_contract") != observation_contract(task):
        raise ValueError("V49 observation contract mismatch")
    agent.load_state_dict(source["agent"], strict=True)
    # Retain V48's trained router, but restart the failed geometry encoder from
    # the validated V41 representation before collecting student-induced states.
    agent.geometry_encoder.load_state_dict(agent.v47.v41.base.encoder.state_dict(), strict=True)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)
    for parameter in agent.geometry_encoder.parameters():
        parameter.requires_grad_(True)
    parameters = list(agent.geometry_encoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"], eps=1e-5)
    updates = int(task["onpolicy_updates"]); transitions = updates * n
    if transitions != int(task["total_timesteps"]):
        raise ValueError("V49 transition budget mismatch")
    history = []; resets = 0; agent.train(); agent.v47.eval(); agent.geometry_router.eval()
    for update in range(updates):
        rgb, proprio, _ = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False), False
        )
        mode = GEOMETRIC_MODES[update % len(GEOMETRIC_MODES)]
        transformed = apply_visual_perturbation(rgb, mode)
        with torch.no_grad():
            target = agent.v47.v41.encode(rgb)
            target_action = action_from_latent(agent, target, proprio)
        latent = agent.geometry_latent(transformed)
        student_action = action_from_latent(agent, latent, proprio)
        feature_loss = F.smooth_l1_loss(latent, target, beta=0.1)
        cosine_loss = (1 - F.cosine_similarity(latent, target, dim=1)).mean()
        action_loss = F.mse_loss(student_action, target_action)
        loss = (task["feature_weight"] * feature_loss + task["cosine_weight"] * cosine_loss
                + task["action_weight"] * action_loss)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.no_grad():
            observation, _, terminated, truncated, _ = environment.step(student_action.detach())
            if bool(torch.logical_or(terminated, truncated).any()):
                resets += 1; observation, _ = environment.reset(seed=seed + 70_000_000 + resets)
        item = {"loss": float(loss.detach()), "feature_loss": float(feature_loss.detach()),
                "cosine_loss": float(cosine_loss.detach()), "action_loss": float(action_loss.detach()),
                "mode": mode}
        history.append(item)
        if (update + 1) % int(config["log_freq"]) == 0:
            recent = history[-int(config["log_freq"]):]
            payload = {"update": update + 1, "simulator_transitions": (update + 1) * n,
                       "mode": mode}
            for key in ("loss", "feature_loss", "cosine_loss", "action_loss"):
                payload[key] = float(np.mean([record[key] for record in recent]))
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(payload) + "\n")
    environment.close(); recent = history[-100:]
    metrics = {f"mean_last_100_{key}": float(np.mean([record[key] for record in recent]))
               for key in ("loss", "feature_loss", "cosine_loss", "action_loss")}
    hashes = {"trainer": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "agent": hashlib.sha256(Path(__file__).with_name("v48_geometry_expert_agent.py").read_bytes()).hexdigest(),
              "environment": hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()}
    checkpoint = {"schema_version": 1, "training_protocol": "onpolicy_geometry_expert_v19",
                  "observation_contract": observation_contract(task), "source_sha256": hashes,
                  "task": task, "agent": agent.state_dict(), "optimizer": optimizer.state_dict(),
                  "iteration": updates, "global_step": transitions,
                  "best_score": -metrics["mean_last_100_loss"], "best_metrics": metrics}
    atomic_save(checkpoint, run_dir / "best.pt"); atomic_save(checkpoint, run_dir / "latest.pt")
    initialization = cumulative_source_interactions(json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text()))
    completion = {"schema_version": 1, "training_protocol": "onpolicy_geometry_expert_v19",
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
