#!/usr/bin/env python3
"""Closed-loop V4 evaluation with no hand-written mechanism routing rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
import atr.envs.learned_recovery_v4_ood  # noqa: F401
from atr.policies.causal_option_router import (
    CausalOptionRouter, StaticOptionRouter, UnstructuredOptionGRU,
    current_centered_sequence,
)
from collect_v4_option_router_data import POSE_KEYS, SNAPSHOTS, extract_features
from evaluate_v19_on_v4 import CONDITIONS, SEEDS
from train_manipulation_ppo import Agent as StateAgent
from train_v4_permanent_visual_dagger import reconstruct_v4_state_teacher_observation
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent, env_kwargs, extract_observation, observation_contract,
    privileged_aux_dim, reconstruct_state_teacher_observation, select_task,
)


def load_router(checkpoint_path: Path, metadata_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    if checkpoint["feature_metadata_sha256"] != metadata_hash:
        raise ValueError("router feature metadata hash mismatch")
    name = checkpoint["model"]
    if name == "causal_gru": model = CausalOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    elif name == "static_mlp": model = StaticOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"])
    elif name == "unstructured_gru": model = UnstructuredOptionGRU(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    else: raise ValueError(f"unknown router model: {name}")
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def router_probability(model, history, geometry_dim=0):
    sequence = torch.stack(history, dim=1)
    length = torch.full(
        (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
        device=sequence.device,
    )
    sequence = current_centered_sequence(sequence, length, geometry_dim)
    output = model(sequence)
    logp = output.option_log_probability if hasattr(output, "option_log_probability") else output
    return logp.exp()


def retreat_action(observation, initial_qpos, action_shape):
    action = torch.zeros((initial_qpos.shape[0],) + action_shape, device=initial_qpos.device)
    arm_width = min(7, action.shape[1], initial_qpos.shape[1])
    delta = initial_qpos[:, :arm_width] - observation["agent"]["qpos"][:, :arm_width]
    action[:, :arm_width] = (8.0 * delta).clamp(-1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--router-metadata", default="results/router/v4_option_prefixes_train_v1.json")
    parser.add_argument("--output-dir", default="results/v4_learned_router_development")
    parser.add_argument("--config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default="results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19")
    parser.add_argument("--permanent-state-checkpoint", default="results/manipulation_ppo/learned_recovery_v4_delayed_permanent_transfer/delayed_permanent_state_transfer/seed_9351/delayed_frozen_iter24.pt")
    parser.add_argument("--reverse-state-checkpoint", default="results/learned_recovery_v4/learned_recovery_v4_reverse_state_pilot/reverse_ejection_state_specialist/seed_9351/reverse_frozen_iter424.pt")
    parser.add_argument("--forward-state-checkpoint", default="results/learned_recovery/learned_recovery_ppo_v11_strict_removal/event_reward_strict_removal_state_ppo/seed_9351/best.pt")
    parser.add_argument(
        "--nominal-state-checkpoint",
        help=(
            "Optional input-matched state PPO shared by nominal execution and "
            "temporary recovery after clearance. When set, no visual nominal "
            "policy is loaded."
        ),
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed-base", type=int, default=310_000_000)
    parser.add_argument("--confirmation-steps", type=int, default=2)
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--onset-step", type=int, default=0)
    parser.add_argument("--return-delay", type=int, default=30)
    parser.add_argument("--control-delay", type=int, default=0)
    parser.add_argument(
        "--safe-hold-until-step", type=int, default=0,
        help=(
            "Execute the defer/retreat option through this step while the router "
            "still selects nominal. An observed event can override the hold early."
        ),
    )
    parser.add_argument(
        "--release-safe-hold-on-confirmed-nominal",
        action="store_true",
        help=(
            "Release the initial defer as soon as the causal router confirms "
            "nominal execution; later observed events can still revise it."
        ),
    )
    parser.add_argument("--env-id", default="LearnedRecovery-v4")
    parser.add_argument("--visual-domain-profile")
    parser.add_argument("--fixed-option", type=int, choices=range(6))
    parser.add_argument("--fixed-option-start-step", type=int, default=1)
    parser.add_argument("--nominal-ensemble", action="store_true")
    parser.add_argument("--nominal-ensemble-reduction", choices=("mean", "median"), default="mean")
    parser.add_argument("--nominal-policy-index", type=int, choices=(0, 1, 2))
    parser.add_argument("--temporary-policy-index", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    combinations = [(seed_index, condition) for seed_index in range(3) for condition in CONDITIONS]
    seed_index, condition = combinations[args.task_index]
    config = json.loads(Path(args.config).read_text()); task, _ = select_task(config, seed_index)
    if int(task["seed"]) != SEEDS[seed_index]: raise ValueError("unexpected training seed order")
    kwargs = env_kwargs(task, evaluation=True)
    kwargs.update({
        "intervention_probability": 0.0 if condition == "nominal" else 1.0,
        "intervention_types": (("ejection",) if condition == "nominal" else (condition,)),
        "onset_step_range": (args.onset_step, args.onset_step),
        "intervention_force": 6.0 * args.force_scale, "intervention_steps": 24,
        "blocker_force": 4.0 * args.force_scale, "blocker_return_force": 5.0 * args.force_scale,
        "blocker_return_delay_steps": args.return_delay, "control_delay_steps": args.control_delay,
    })
    if args.env_id == "LearnedRecovery-v4-OOD":
        if not args.visual_domain_profile: raise ValueError("OOD environment requires a profile")
        kwargs["visual_domain_profile"] = args.visual_domain_profile
    env = gym.make(args.env_id, num_envs=args.num_envs, reconfiguration_freq=1, max_episode_steps=args.steps, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict): env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=False)
    device = torch.device("cuda")
    router, router_checkpoint = load_router(Path(args.router_checkpoint), Path(args.router_metadata), device)
    current_centered_geometry_dim = int(router_checkpoint.get("current_centered_geometry_dim", 0))
    router_metadata = json.loads(Path(args.router_metadata).read_text())
    router_horizon = max(int(step) for step in router_metadata["snapshots"])
    first, _ = env.reset(seed=args.seed_base + int(task["seed"]) * 100_000)
    rgb, proprio, critic = extract_observation(first, task["asymmetric_critic"], task.get("actor_tcp_pose", False), task.get("actor_goal_progress", False))
    nominal_indices = list(range(3)) if args.nominal_ensemble else [
        args.nominal_policy_index if args.nominal_policy_index is not None else seed_index
    ]
    temporary_index = args.temporary_policy_index if args.temporary_policy_index is not None else seed_index
    loaded_indices = list(dict.fromkeys([*nominal_indices, temporary_index]))
    nominal_tasks = (
        [] if args.nominal_state_checkpoint
        else [select_task(config, index)[0] for index in loaded_indices]
    )
    nominal_agents = []
    for nominal_task in nominal_tasks:
        member_rgb, member_proprio, member_critic = extract_observation(
            first, nominal_task["asymmetric_critic"],
            nominal_task.get("actor_tcp_pose", False),
            nominal_task.get("actor_goal_progress", False),
        )
        agent = VisualAgent(
            nominal_task["image_size"], member_proprio.shape[1], member_critic.shape[1],
            int(np.prod(env.single_action_space.shape)), nominal_task["asymmetric_critic"],
            nominal_task.get("augmentation_pad", 0), privileged_aux_dim(nominal_task),
            nominal_task.get("actor_learned_goal_progress", False),
        ).cuda()
        path = Path(args.checkpoint_root) / nominal_task["method"] / f"seed_{nominal_task['seed']}" / "best.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("observation_contract") != observation_contract(nominal_task):
            raise ValueError("nominal checkpoint observation contract mismatch")
        agent.load_state_dict(checkpoint["agent"]); agent.eval()
        nominal_agents.append((agent, nominal_task))
    action_dim = int(np.prod(env.single_action_space.shape))
    v4_state = reconstruct_v4_state_teacher_observation(first); v3_state = reconstruct_state_teacher_observation(first)
    state_specs = {
        "permanent": (Path(args.permanent_state_checkpoint), v4_state.shape[1]),
        "reverse": (Path(args.reverse_state_checkpoint), v4_state.shape[1]),
        "forward": (Path(args.forward_state_checkpoint), v3_state.shape[1]),
    }
    if args.nominal_state_checkpoint:
        state_specs["nominal"] = (
            Path(args.nominal_state_checkpoint), v4_state.shape[1],
        )
    state_agents = {}
    for name, (path, width) in state_specs.items():
        agent = StateAgent(width, action_dim).cuda(); agent.load_state_dict(torch.load(path, map_location=device, weights_only=False)["agent"], strict=True); agent.eval(); state_agents[name] = agent
    threshold = float(router_checkpoint["calibration"]["threshold"])
    class_thresholds = torch.tensor(
        router_checkpoint["calibration"].get(
            "class_thresholds_99_precision", [threshold] * 6,
        ), device=device,
    )
    successes = safe_successes = violations = decisions = correct_decisions = abstentions = 0
    option_histogram = torch.zeros(6, dtype=torch.long, device=device)
    try:
        for offset in range(0, args.episodes, args.num_envs):
            obs, _ = env.reset(seed=args.seed_base + int(task["seed"]) * 100_000 + offset)
            positions = torch.stack([obs["extra"][key][:, :3] for key in POSE_KEYS], dim=1)
            initial_positions = positions.clone(); previous_positions = positions.clone()
            initial_tcp = obs["extra"]["tcp_pose"][:, :3].clone(); initial_qpos = obs["agent"]["qpos"].clone()
            history = []; candidate = torch.full((args.num_envs,), 5, dtype=torch.long, device=device)
            candidate_count = torch.zeros(args.num_envs, dtype=torch.long, device=device)
            # Optimistic nominal execution is the safe default for an episode
            # with no observed event.  The learned readiness/event posterior
            # can explicitly defer at the first query or hand off to recovery.
            selected_option = torch.zeros_like(candidate)
            nominal_confirmed = torch.zeros(
                args.num_envs, dtype=torch.bool, device=device,
            )
            decision_locked = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            success = torch.zeros(args.num_envs, dtype=torch.bool, device=device); violation = torch.zeros_like(success)
            for step in range(1, args.steps + 1):
                if args.fixed_option is None and step <= router_horizon:
                    feature, positions = extract_features(
                        obs, initial_positions, previous_positions, initial_tcp,
                        initial_qpos, step, router_horizon,
                    )
                    previous_positions = positions
                    history.append(feature)
                if args.fixed_option is None and step in SNAPSHOTS:
                    probability = router_probability(
                        router, history, current_centered_geometry_dim,
                    )
                    confidence, proposed = probability.max(1)
                    proposed = torch.where(
                        confidence >= class_thresholds[proposed], proposed,
                        torch.full_like(proposed, 5),
                    )
                    same = proposed == candidate
                    candidate_count = torch.where(same, candidate_count + 1, torch.ones_like(candidate_count))
                    candidate = proposed
                    confirmed = candidate_count >= args.confirmation_steps
                    accepted = confirmed & ~decision_locked
                    selected_option = torch.where(accepted, candidate, selected_option)
                    nominal_confirmed |= accepted & (candidate == 0)
                    # Specialist hand-offs are irreversible. Nominal/defer
                    # remain revisable so delayed events can still be routed.
                    irreversible = torch.isin(
                        candidate, torch.tensor((1, 2, 3), device=device)
                    )
                    decision_locked |= accepted & irreversible
                elif args.fixed_option is not None:
                    selected_option.fill_(
                        args.fixed_option if step >= args.fixed_option_start_step else 5
                    )
                effective_option = torch.where(
                    (selected_option == 0)
                    & (step <= args.safe_hold_until_step)
                    & ~(
                        args.release_safe_hold_on_confirmed_nominal
                        & nominal_confirmed
                    ),
                    torch.full_like(selected_option, 5),
                    selected_option,
                )
                abstentions += int((effective_option == 5).sum())
                v4_state = reconstruct_v4_state_teacher_observation(obs); v3_state = reconstruct_state_teacher_observation(obs)
                if args.nominal_state_checkpoint:
                    nominal_action = state_agents["nominal"].get_action(
                        v4_state, deterministic=True,
                    ).clamp(-1, 1)
                    temporary_action = nominal_action
                else:
                    nominal_actions = []
                    for agent, nominal_task in nominal_agents:
                        rgb, proprio, _ = extract_observation(
                            obs, nominal_task["asymmetric_critic"],
                            nominal_task.get("actor_tcp_pose", False),
                            nominal_task.get("actor_goal_progress", False),
                        )
                        latent = agent.encode(rgb)
                        native = torch.sigmoid(agent.goal_progress_predictor(latent))
                        nominal_actions.append(torch.tanh(agent.actor(torch.cat((latent, proprio, native), dim=1))))
                    action_by_index = dict(zip(loaded_indices, nominal_actions))
                    nominal_stack = torch.stack([action_by_index[index] for index in nominal_indices])
                    nominal_action = (
                        nominal_stack.median(0).values
                        if args.nominal_ensemble_reduction == "median"
                        else nominal_stack.mean(0)
                    )
                    temporary_action = action_by_index[temporary_index]
                actions = (
                    nominal_action,
                    state_agents["forward"].get_action(v3_state, deterministic=True).clamp(-1, 1),
                    state_agents["reverse"].get_action(v4_state, deterministic=True).clamp(-1, 1),
                    state_agents["permanent"].get_action(v4_state, deterministic=True).clamp(-1, 1),
                    temporary_action,
                    retreat_action(obs, initial_qpos, env.single_action_space.shape),
                )
                stacked = torch.stack(actions, dim=1)
                action = stacked[torch.arange(args.num_envs, device=device), effective_option]
                option_histogram += torch.bincount(effective_option, minlength=6)
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool(); violation |= info["constraint_violated"].bool()
            successes += int(success.sum()); safe_successes += int((success & ~violation).sum()); violations += int(violation.sum())
            truth = {"nominal": 0, "ejection": 1, "reverse_ejection": 2, "permanent_block": 3, "temporary_block": 4}[condition]
            decisions += args.num_envs; correct_decisions += int((selected_option == truth).sum())
    finally: env.close()
    result = {
        "schema_version": 1, "method": router_checkpoint["model"], "router_seed": router_checkpoint["seed"],
        "training_policy_seed": int(task["seed"]), "condition": condition, "episodes": args.episodes,
        "successes": successes, "success_rate": successes / args.episodes,
        "safe_successes": safe_successes, "safe_success_rate": safe_successes / args.episodes,
        "violations": violations, "violation_rate": violations / args.episodes,
        "final_option_accuracy": correct_decisions / decisions, "confidence_threshold": threshold,
        "class_thresholds_99_precision": class_thresholds.cpu().tolist(),
        "confirmation_steps": args.confirmation_steps, "option_step_histogram": option_histogram.cpu().tolist(),
        "abstention_step_rate": abstentions / (args.episodes * args.steps),
        "seed_base": args.seed_base, "force_scale": args.force_scale, "onset_step": args.onset_step,
        "safe_hold_until_step": args.safe_hold_until_step,
        "release_safe_hold_on_confirmed_nominal": (
            args.release_safe_hold_on_confirmed_nominal
        ),
        "return_delay": args.return_delay, "control_delay": args.control_delay,
        "environment": args.env_id, "visual_domain_profile": args.visual_domain_profile or "nominal",
        "fixed_option": args.fixed_option,
        "fixed_option_start_step": args.fixed_option_start_step,
        "nominal_policy_type": "state_ppo" if args.nominal_state_checkpoint else "visual_ppo",
        "nominal_state_checkpoint": args.nominal_state_checkpoint,
        "nominal_state_checkpoint_sha256": (
            hashlib.sha256(Path(args.nominal_state_checkpoint).read_bytes()).hexdigest()
            if args.nominal_state_checkpoint else None
        ),
        "nominal_ensemble": args.nominal_ensemble if not args.nominal_state_checkpoint else False,
        "nominal_ensemble_reduction": args.nominal_ensemble_reduction,
        "nominal_ensemble_seeds": (
            [int(select_task(config, index)[0]["seed"]) for index in nominal_indices]
            if not args.nominal_state_checkpoint else []
        ),
        "temporary_policy_seed": (
            int(select_task(config, temporary_index)[0]["seed"])
            if not args.nominal_state_checkpoint else None
        ),
        "router_checkpoint": args.router_checkpoint,
        "router_checkpoint_sha256": hashlib.sha256(Path(args.router_checkpoint).read_bytes()).hexdigest(),
        "feature_metadata_sha256": router_checkpoint["feature_metadata_sha256"],
        "current_centered_geometry_dim": current_centered_geometry_dim,
        "heldout_option": router_checkpoint.get("heldout_option"),
        "forbidden_runtime_inputs": ["mechanism ID", "intervention target", "critic_goal_resolved", "future observation"],
    }
    output = Path(args.output_dir) / f"router{router_checkpoint['seed']}_policy{task['seed']}_{condition}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
