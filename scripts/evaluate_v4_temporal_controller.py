#!/usr/bin/env python3
"""Closed-loop V4 evaluation of temporal DINO feasibility routing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import joblib
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
import atr.envs.learned_recovery_v4_ood  # noqa: F401
from evaluate_v19_on_v4 import CONDITIONS, SEEDS
from probe_v4_temporal_feasibility import goal_condition, visual_deltas
from train_manipulation_ppo import Agent as StateAgent
from train_v4_permanent_visual_dagger import reconstruct_v4_state_teacher_observation
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent, env_kwargs, extract_observation, observation_contract,
    privileged_aux_dim, reconstruct_state_teacher_observation, select_task,
)


ONSET_HORIZON = 4
DEFER_HORIZON = 48
MOTION_THRESHOLD = 0.005
BLOCKER_ENGAGED_X = 0.22
BLOCKAGE_DECISION_HORIZON = 36
BLOCKER_CLEARED_X = 0.42


@torch.inference_mode()
def routed_action(agent, rgb, proprio, unavailable):
    latent = agent.encode(rgb)
    native = torch.sigmoid(agent.goal_progress_predictor(latent))
    effective = torch.maximum(native, unavailable.float())
    return torch.tanh(agent.actor(torch.cat((latent, proprio, effective), dim=1))), native


def retreat_action(observation, initial_qpos, action_shape):
    action = torch.zeros(
        (initial_qpos.shape[0],) + action_shape,
        device=initial_qpos.device, dtype=torch.float32,
    )
    arm_width = min(7, action.shape[1], initial_qpos.shape[1])
    delta = initial_qpos[:, :arm_width] - observation["agent"]["qpos"][:, :arm_width]
    action[:, :arm_width] = (8.0 * delta).clamp(-1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default="results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19")
    parser.add_argument("--classifier", default="results/models/v4_temporal_dinov2_two_stage_v7.joblib")
    parser.add_argument("--classifier-metadata", default="results/models/v4_temporal_dinov2_two_stage_v7.json")
    parser.add_argument("--output-dir", default="results/v4_temporal_controller_v16_hybrid_state")
    parser.add_argument(
        "--permanent-state-checkpoint",
        default=(
            "results/manipulation_ppo/learned_recovery_v4_delayed_permanent_transfer/"
            "delayed_permanent_state_transfer/seed_9351/delayed_frozen_iter24.pt"
        ),
    )
    parser.add_argument(
        "--reverse-state-checkpoint",
        default=(
            "results/learned_recovery_v4/learned_recovery_v4_reverse_state_pilot/"
            "reverse_ejection_state_specialist/seed_9351/reverse_frozen_iter424.pt"
        ),
    )
    parser.add_argument("--forward-state-checkpoint", default=(
        "results/learned_recovery/learned_recovery_ppo_v11_strict_removal/"
        "event_reward_strict_removal_state_ppo/seed_9351/best.pt"
    ))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--env-id", default="LearnedRecovery-v4")
    parser.add_argument("--visual-domain-profile")
    parser.add_argument("--seed-base", type=int, default=190_000_000)
    args = parser.parse_args()
    combinations = [(seed_index, condition) for seed_index in range(3) for condition in CONDITIONS]
    seed_index, condition = combinations[args.task_index]
    config = json.loads(Path(args.config).read_text())
    task, _ = select_task(config, seed_index)
    seed = int(task["seed"])
    if seed != SEEDS[seed_index]:
        raise ValueError("unexpected training seed order")

    kwargs = env_kwargs(task, evaluation=True)
    kwargs.update({
        "intervention_probability": 0.0 if condition == "nominal" else 1.0,
        "intervention_types": (("ejection",) if condition == "nominal" else (condition,)),
        "onset_step_range": (0, 0), "intervention_force": 6.0,
        "intervention_steps": 24, "blocker_force": 4.0,
        "blocker_return_force": 5.0, "blocker_return_delay_steps": 30,
    })
    if args.env_id == "LearnedRecovery-v4-OOD":
        if not args.visual_domain_profile:
            raise ValueError("OOD evaluation requires --visual-domain-profile")
        kwargs["visual_domain_profile"] = args.visual_domain_profile
    env = gym.make(
        args.env_id, num_envs=args.num_envs, reconfiguration_freq=1,
        max_episode_steps=args.steps, **kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=False)

    checkpoint_path = Path(args.checkpoint_root) / task["method"] / f"seed_{seed}" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    classifier_path = Path(args.classifier)
    metadata = json.loads(Path(args.classifier_metadata).read_text())
    if hashlib.sha256(classifier_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("classifier hash mismatch")
    classifiers = joblib.load(classifier_path)
    late_horizon = int(metadata["horizon"])
    snapshots = tuple(step for step in (1, 4, 8, 16, 32, 48) if step <= late_horizon)
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14", verbose=False,
    ).eval().cuda()

    first, _ = env.reset(seed=args.seed_base + seed * 100_000)
    rgb, proprio, critic = extract_observation(
        first, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1],
        int(np.prod(env.single_action_space.shape)), task["asymmetric_critic"],
        task.get("augmentation_pad", 0), privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    if checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("checkpoint observation contract mismatch")
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    action_dim = int(np.prod(env.single_action_space.shape))
    v4_state = reconstruct_v4_state_teacher_observation(first)
    v3_state = reconstruct_state_teacher_observation(first)
    state_specs = {
        "permanent": (Path(args.permanent_state_checkpoint), v4_state.shape[1]),
        "reverse": (Path(args.reverse_state_checkpoint), v4_state.shape[1]),
        "forward": (Path(args.forward_state_checkpoint), v3_state.shape[1]),
    }
    state_agents = {}
    for name, (path, width) in state_specs.items():
        state_checkpoint = torch.load(path, map_location="cuda", weights_only=False)
        state_agent = StateAgent(width, action_dim).cuda()
        state_agent.load_state_dict(state_checkpoint["agent"], strict=True)
        state_agent.eval(); state_agents[name] = state_agent
    successes = violations = classifier_correct = classifier_queries = 0
    onset_correct = onset_total = 0
    predictions = []
    try:
        for offset in range(0, args.episodes, args.num_envs):
            obs, _ = env.reset(seed=args.seed_base + seed * 100_000 + offset)
            reference = obs["sensor_data"]["base_camera"]["rgb"].clone()
            initial_qpos = obs["agent"]["qpos"].clone()
            tracked_pose_keys = (
                "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
                "critic_red_reverse_sweeper_pose", "critic_blue_reverse_sweeper_pose",
                "critic_red_goal_blocker_pose", "critic_blue_goal_blocker_pose",
            )
            initial_actor_x = torch.stack(
                [obs["extra"][key][:, 0] for key in tracked_pose_keys], dim=1,
            )
            history = []
            onset_history = []
            unavailable = torch.zeros((args.num_envs, 2), dtype=torch.bool, device="cuda")
            event_family = torch.full(
                (args.num_envs,), -1, dtype=torch.long, device="cuda"
            )
            persistent_blockage = torch.zeros(
                args.num_envs, dtype=torch.bool, device="cuda"
            )
            # 0 nominal, 1 forward ejection, 2 reverse ejection, 3 blockage.
            # The router observes motion only; it never reads the simulator's
            # intervention mechanism or intervention target.
            physical_family = torch.full(
                (args.num_envs,), -1, dtype=torch.long, device="cuda"
            )
            previous_blocker_x = initial_actor_x[:, 4:].amin(dim=1)
            temporary_returning = torch.zeros(
                args.num_envs, dtype=torch.bool, device="cuda"
            )
            success = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
            violation = torch.zeros_like(success)
            for step in range(1, args.steps + 1):
                actor_x = torch.stack(
                    [obs["extra"][key][:, 0] for key in tracked_pose_keys], dim=1,
                )
                displacement = (actor_x - initial_actor_x).abs()
                newly_observed = physical_family < 0
                forward_moved = displacement[:, :2].amax(dim=1) > MOTION_THRESHOLD
                reverse_moved = displacement[:, 2:4].amax(dim=1) > MOTION_THRESHOLD
                blocker_moved = displacement[:, 4:].amax(dim=1) > MOTION_THRESHOLD
                physical_family = torch.where(
                    newly_observed & forward_moved,
                    torch.ones_like(physical_family), physical_family,
                )
                physical_family = torch.where(
                    newly_observed & reverse_moved,
                    torch.full_like(physical_family, 2), physical_family,
                )
                physical_family = torch.where(
                    newly_observed & blocker_moved,
                    torch.full_like(physical_family, 3), physical_family,
                )
                if step >= 2:
                    physical_family = torch.where(
                        physical_family < 0,
                        torch.zeros_like(physical_family), physical_family,
                    )
                blocker_x = actor_x[:, 4:].amin(dim=1)
                temporary_returning |= (
                    (physical_family == 3)
                    & (step > 30)
                    & (blocker_x > previous_blocker_x + 0.001)
                )
                previous_blocker_x = blocker_x
                rgb, proprio, _ = extract_observation(
                    obs, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                    task.get("actor_goal_progress", False),
                )
                if step in snapshots:
                    deltas = visual_deltas(backbone, reference, rgb)
                    history.append(deltas["dinov2"])
                    onset_history.append(deltas["dinov2_axis"])
                if step == ONSET_HORIZON:
                    onset_index = snapshots.index(ONSET_HORIZON)
                    early_query = goal_condition(torch.cat(onset_history[:onset_index + 1], dim=1)).reshape(
                        args.num_envs * 2, -1
                    )
                    episode_query = early_query.reshape(
                        args.num_envs, 2, -1
                    ).reshape(args.num_envs, -1)
                    event_family = torch.from_numpy(
                        classifiers["onset"].predict(episode_query.cpu().numpy())
                    ).to("cuda")
                    true_family = torch.full_like(event_family, 2)
                    if condition == "nominal":
                        true_family[:] = 0
                    elif condition in ("ejection", "reverse_ejection"):
                        true_family[:] = 1
                    onset_correct += int((event_family == true_family).sum())
                    onset_total += args.num_envs
                if step == late_horizon:
                    query = goal_condition(torch.cat(history, dim=1)).reshape(args.num_envs * 2, -1)
                    probability = classifiers["late_feasibility"].predict_proba(
                        query.cpu().numpy()
                    )[:, 1]
                    late_unavailable = torch.from_numpy(
                        probability >= 0.5
                    ).to("cuda").reshape(args.num_envs, 2)
                    blockage = event_family == 2
                    persistent_blockage = blockage & late_unavailable.any(dim=1)
                    unavailable = torch.where(
                        persistent_blockage[:, None], late_unavailable, unavailable
                    )
                    target = (
                        obs["extra"]["critic_goal_resolved"].bool()
                        & ~obs["extra"]["goal_progress"].bool()
                    )
                    classifier_correct += int(
                        ((late_unavailable == target) & blockage[:, None]).sum()
                    )
                    classifier_queries += int(blockage.sum()) * 2
                    predictions.extend(probability.tolist())
                action, _ = routed_action(agent, rgb, proprio, unavailable)
                blocker_target = displacement[:, 4:].argmax(dim=1)
                provisional_blocked = (physical_family == 3) & ~temporary_returning
                selected_blocker_x = actor_x[
                    torch.arange(args.num_envs, device="cuda"), blocker_target + 4
                ]
                blocker_engaged_observed = selected_blocker_x < BLOCKER_ENGAGED_X
                temporary_cleared_observed = (
                    temporary_returning & (selected_blocker_x > BLOCKER_CLEARED_X)
                )
                v4_state = reconstruct_v4_state_teacher_observation(obs)
                v3_state = reconstruct_state_teacher_observation(obs)
                permanent_action = state_agents["permanent"].get_action(
                    v4_state, deterministic=True,
                ).clamp(-1, 1)
                reverse_action = state_agents["reverse"].get_action(
                    v4_state, deterministic=True,
                ).clamp(-1, 1)
                forward_action = state_agents["forward"].get_action(
                    v3_state, deterministic=True,
                ).clamp(-1, 1)
                action = torch.where(
                    (physical_family == 1)[:, None], forward_action, action
                )
                action = torch.where(
                    (physical_family == 2)[:, None], reverse_action, action
                )
                action = torch.where(
                    (
                        (physical_family == 3)
                        & ~temporary_returning
                        & blocker_engaged_observed
                        & (step > BLOCKAGE_DECISION_HORIZON)
                    )[:, None],
                    permanent_action, action,
                )
                retreat = retreat_action(
                    obs, initial_qpos, env.single_action_space.shape,
                )
                action = torch.where(
                    (
                        (physical_family < 0)
                        | (
                            (physical_family == 3)
                            & ~temporary_cleared_observed
                            & (
                                temporary_returning
                                | ~blocker_engaged_observed
                                | (step <= BLOCKAGE_DECISION_HORIZON)
                            )
                        )
                    )[:, None],
                    retreat, action,
                )
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool()
                violation |= info["constraint_violated"].bool()
            successes += int(success.sum())
            violations += int(violation.sum())
    finally:
        env.close()
    result = {
        "schema_version": 21, "method": "exact_handoff_ood_router",
        "seed": seed, "condition": condition, "episodes": args.episodes,
        "episode_horizon": args.steps,
        "environment": args.env_id,
        "visual_domain_profile": args.visual_domain_profile or "nominal",
        "successes": successes, "success_rate": successes / args.episodes,
        "violations": violations, "violation_rate": violations / args.episodes,
        "onset_family_accuracy": onset_correct / onset_total,
        "classifier_query_accuracy": (
            classifier_correct / classifier_queries if classifier_queries else None
        ),
        "classifier_probability_mean": float(np.mean(predictions)),
        "classifier_sha256": metadata["model_sha256"],
        "late_horizon": late_horizon,
        "checkpoint": str(checkpoint_path),
        "state_specialists": {
            name: {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, (path, _) in state_specs.items()
        },
    }
    output = Path(args.output_dir) / f"seed_{seed}_{condition}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
