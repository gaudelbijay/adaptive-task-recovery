#!/usr/bin/env python3
"""Fit and freeze the canonical V4 temporal feasibility classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from probe_v4_temporal_feasibility import goal_condition, make_env, visual_deltas
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent, extract_observation, privileged_aux_dim, select_task,
)


TRAINING_KINDS = ("nominal", "ejection", "permanent_block", "temporary_block")
ONSET_HORIZON = 4
FAMILY = "dinov2"
ONSET_FAMILY = "dinov2_axis"


def regularized_fit(features, labels):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.003, max_iter=3000, class_weight="balanced"),
    ).fit(features, labels)


def load_v19_agents(observation, env, *, config_path, checkpoint_root):
    config = json.loads(Path(config_path).read_text())
    agents, tasks = [], []
    for index in range(3):
        task, _ = select_task(config, index)
        _, proprio, critic = extract_observation(
            observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
            task.get("actor_goal_progress", False),
        )
        agent = VisualAgent(
            task["image_size"], proprio.shape[1], critic.shape[1],
            int(np.prod(env.single_action_space.shape)), task["asymmetric_critic"],
            task.get("augmentation_pad", 0), privileged_aux_dim(task),
            task.get("actor_learned_goal_progress", False),
        ).cuda()
        checkpoint = Path(checkpoint_root) / task["method"] / f"seed_{task['seed']}" / "best.pt"
        agent.load_state_dict(torch.load(
            checkpoint, map_location="cuda", weights_only=False,
        )["agent"])
        agent.eval()
        agents.append(agent)
        tasks.append(task)
    return agents, tasks


@torch.inference_mode()
def collect_onpolicy_onset(
    backbone, *, kind, seed_base, batches, num_envs, agents, tasks,
):
    env = make_env(num_envs, kind, "nominal")
    episode_rows, query_rows, target_rows = [], [], []
    try:
        for batch in range(batches):
            obs, _ = env.reset(seed=seed_base + batch * num_envs)
            reference = obs["sensor_data"]["base_camera"]["rgb"].clone()
            history = []
            agent = agents[batch % len(agents)]
            task = tasks[batch % len(tasks)]
            for step in range(1, ONSET_HORIZON + 1):
                rgb, proprio, _ = extract_observation(
                    obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                    task.get("actor_goal_progress", False),
                )
                if step in (1, ONSET_HORIZON):
                    history.append(visual_deltas(
                        backbone, reference, rgb,
                    )[ONSET_FAMILY])
                obs, _, _, _, _ = env.step(
                    agent.get_action(rgb, proprio, deterministic=True)
                )
            queries = goal_condition(torch.cat(history, dim=1))
            episode_rows.append(queries.reshape(num_envs, -1).cpu().numpy())
            query_rows.append(queries.reshape(num_envs * 2, -1).cpu().numpy())
            blockers = torch.stack((
                obs["extra"]["critic_red_goal_blocker_pose"][:, 0],
                obs["extra"]["critic_blue_goal_blocker_pose"][:, 0],
            ), dim=1)
            target_rows.append(torch.nn.functional.one_hot(
                blockers.argmin(dim=1), 2,
            ).reshape(-1).cpu().numpy().astype(np.int64))
    finally:
        env.close()
    return (
        np.concatenate(episode_rows), np.concatenate(query_rows),
        np.concatenate(target_rows),
    )


@torch.inference_mode()
def collect_deferred_late(
    backbone, *, kind, seed_base, batches, num_envs, agents, tasks, horizon,
):
    env = make_env(num_envs, kind, "nominal")
    feature_rows, label_rows = [], []
    snapshots = tuple(step for step in (1, 4, 8, 16, 32, 48) if step <= horizon)
    try:
        for batch in range(batches):
            obs, _ = env.reset(seed=seed_base + batch * num_envs)
            reference = obs["sensor_data"]["base_camera"]["rgb"].clone()
            history = []
            agent = agents[batch % len(agents)]
            task = tasks[batch % len(tasks)]
            for step in range(1, horizon + 1):
                rgb, proprio, _ = extract_observation(
                    obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                    task.get("actor_goal_progress", False),
                )
                if step in snapshots:
                    history.append(visual_deltas(backbone, reference, rgb)[FAMILY])
                action = (
                    agent.get_action(rgb, proprio, deterministic=True)
                    if step < ONSET_HORIZON
                    else torch.zeros(
                        (num_envs,) + env.single_action_space.shape,
                        device=rgb.device, dtype=torch.float32,
                    )
                )
                obs, _, _, _, _ = env.step(action)
            feature_rows.append(goal_condition(torch.cat(history, dim=1)).reshape(
                num_envs * 2, -1,
            ).cpu().numpy())
            unavailable = (
                obs["extra"]["critic_goal_resolved"].bool()
                & ~obs["extra"]["goal_progress"].bool()
            )
            label_rows.append(unavailable.reshape(-1).cpu().numpy().astype(np.int64))
    finally:
        env.close()
    return np.concatenate(feature_rows), np.concatenate(label_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output", default="results/models/v4_temporal_dinov2_two_stage_v7.joblib")
    parser.add_argument("--metadata-output", default="results/models/v4_temporal_dinov2_two_stage_v7.json")
    parser.add_argument("--train-batches", type=int, default=16)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--late-horizon", type=int, default=32)
    parser.add_argument("--controller-config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default="results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19")
    args = parser.parse_args()
    if args.late_horizon not in (8, 16, 32, 48):
        raise ValueError("late-horizon must be one of 8, 16, 32, or 48")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()
    bootstrap_env = make_env(args.num_envs, "nominal", "nominal")
    try:
        bootstrap_obs, _ = bootstrap_env.reset(seed=150_000_000)
        agents, tasks = load_v19_agents(
            bootstrap_obs, bootstrap_env, config_path=args.controller_config,
            checkpoint_root=args.checkpoint_root,
        )
    finally:
        bootstrap_env.close()
    onset_features, onset_labels = [], []
    blocker_features, blocker_labels = [], []
    for index, kind in enumerate(TRAINING_KINDS):
        episode_rows, query_rows, target_rows = collect_onpolicy_onset(
            backbone, kind=kind, seed_base=171_000_000 + index * 1_000_000,
            batches=args.train_batches, num_envs=args.num_envs,
            agents=agents, tasks=tasks,
        )
        onset_features.append(episode_rows)
        if kind in ("permanent_block", "temporary_block"):
            blocker_features.append(query_rows)
            blocker_labels.append(target_rows)
        onset_class = 0 if kind == "nominal" else (1 if kind == "ejection" else 2)
        onset_labels.append(np.full(
            args.train_batches * args.num_envs, onset_class, dtype=np.int64,
        ))
    late_features, late_labels = [], []
    for index, kind in enumerate(("permanent_block", "temporary_block")):
        rows, target = collect_deferred_late(
            backbone, kind=kind, seed_base=181_000_000 + index * 1_000_000,
            batches=args.train_batches, num_envs=args.num_envs,
            agents=agents, tasks=tasks, horizon=args.late_horizon,
        )
        late_features.append(rows)
        late_labels.append(target)
    late_y = np.concatenate(late_labels)
    artifact = {
        "onset": regularized_fit(np.concatenate(onset_features), np.concatenate(onset_labels)),
        "onset_blocked_goal": regularized_fit(
            np.concatenate(blocker_features), np.concatenate(blocker_labels)
        ),
        "late_feasibility": regularized_fit(
            np.concatenate(late_features), late_y
        ),
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metadata = {
        "schema_version": 6, "family": FAMILY, "onset_family": ONSET_FAMILY,
        "horizon": args.late_horizon,
        "onset_horizon": ONSET_HORIZON, "regularization_c": 0.003,
        "training_kinds": list(TRAINING_KINDS),
        "training_goal_queries": int(len(late_y)),
        "positive_queries": int(late_y.sum()),
        "seed_bases": [171_000_000 + i * 1_000_000 for i in range(len(TRAINING_KINDS))],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "heldout_mechanism": "reverse_ejection",
        "onset_training": "frozen V19 rollouts across three controller seeds",
        "onset_blocked_goal_training": (
            "goal-conditioned physical blocker localization on permanent and "
            "temporary blockage"
        ),
        "late_training": (
            "three V19 actions followed by deferral through step "
            f"{args.late_horizon}"
        ),
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
