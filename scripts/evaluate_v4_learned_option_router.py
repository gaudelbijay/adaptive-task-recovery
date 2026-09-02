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
from atr.policies.option_router import (
    FactorizedOptionRouter, StaticOffsetRouter, StaticOptionRouter,
    UnstructuredOptionGRU, current_centered_sequence,
)
from atr.policies.heuristic_option_router import HeuristicMotionRouter
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
    # "causal_gru" is the frozen persisted identifier for FactorizedOptionRouter;
    # existing checkpoints and gate manifests store it. Not renamed on purpose.
    if name == "causal_gru": model = FactorizedOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    elif name == "static_mlp": model = StaticOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"])
    elif name == "unstructured_gru": model = UnstructuredOptionGRU(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    elif name.startswith("static_offset_"):
        suffix = name.rsplit("_", 1)[1]
        offset = None if suffix == "first" else int(suffix)
        model = StaticOffsetRouter(checkpoint["input_dim"], checkpoint["hidden_dim"], offset)
    elif name == "heuristic_motion":
        # Hand-written V28 baseline. It has no trained parameters; its buffers
        # are feature indices resolved from the same matched metadata.
        model = HeuristicMotionRouter(
            checkpoint["feature_names"], checkpoint["threshold"],
        )
        model.to(device).eval()
        return model, checkpoint
    else: raise ValueError(f"unknown router model: {name}")
    model.load_state_dict(checkpoint["state_dict"]); model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def router_output(model, history, geometry_dim=0):
    sequence = torch.stack(history, dim=1)
    length = torch.full(
        (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
        device=sequence.device,
    )
    sequence = current_centered_sequence(sequence, length, geometry_dim)
    output = model(sequence)
    logp = output.option_log_probability if hasattr(output, "option_log_probability") else output
    return output, logp.exp()


def _render_frame(env):
    image = env.render()
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    return image.astype(np.uint8)


def _write_capture(args, condition, frames, options, resolution, goals, router_checkpoint, success, violation):
    """Write the episode video plus a provenance record beside it."""
    import imageio.v2 as imageio

    from atr.policies.option_router import OPTION_NAMES

    output = Path(args.capture_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    video = Path(f"{stem}_{condition}.mp4")
    imageio.mimsave(video, frames, fps=args.fps_capture, macro_block_size=2)
    record = {
        "schema_version": 1,
        "env_id": args.env_id,
        "condition": condition,
        "method": router_checkpoint["model"],
        "router_seed": router_checkpoint["seed"],
        "router_checkpoint": str(args.router_checkpoint),
        "router_checkpoint_sha256": hashlib.sha256(
            Path(args.router_checkpoint).read_bytes()
        ).hexdigest(),
        "seed_base": args.seed_base,
        "capture_env_index": args.capture_env_index,
        "frames": len(frames),
        "steps": len(options),
        "safe_success": bool(success and not violation),
        "constraint_violated": bool(violation),
        "factorized_sweep_dispatch": bool(args.factorized_sweep_dispatch),
        "selected_option_by_step": options,
        # Scoring stops at first resolution, so frames after this step are not
        # measured and should not be shown as if they were.
        "resolution_step": resolution,
        "goals_at_resolution": goals,
        "option_names": list(OPTION_NAMES),
        "video": str(video),
    }
    Path(f"{stem}_{condition}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"captured {video} ({len(frames)} frames, safe_success={record['safe_success']})")


def retreat_action(observation, initial_qpos, action_shape):
    action = torch.zeros((initial_qpos.shape[0],) + action_shape, device=initial_qpos.device)
    arm_width = min(7, action.shape[1], initial_qpos.shape[1])
    delta = initial_qpos[:, :arm_width] - observation["agent"]["qpos"][:, :arm_width]
    action[:, :arm_width] = (8.0 * delta).clamp(-1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--capture-video",
        help=(
            "Write an mp4 of one environment's episode plus a per-step record "
            "of the router's selected option. Rendering is the only change; "
            "the rollout is the evaluated one."
        ),
    )
    parser.add_argument("--capture-env-index", type=int, default=0)
    parser.add_argument("--fps-capture", type=int, default=20)
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--router-metadata", default="results/router/v4_option_prefixes_train_v1.json")
    parser.add_argument("--output-dir", default="results/v4_learned_router_development")
    parser.add_argument("--config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default="results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19")
    parser.add_argument("--permanent-state-checkpoint", default="results/manipulation_ppo/learned_recovery_v4_delayed_permanent_transfer/delayed_permanent_state_transfer/seed_9351/delayed_frozen_iter24.pt")
    parser.add_argument(
        "--reverse-state-checkpoint", action="append",
        help=(
            "Repeat to form a seed ensemble for the reverse specialist. "
            "The same ensemble is exposed to every router."
        ),
    )
    parser.add_argument(
        "--reverse-ensemble-reduction", choices=("mean", "median"), default="mean",
    )
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
        "--safe-hold-start-step", type=int, default=1,
        help=(
            "First step of the label-free guard window. Steps before it use "
            "ordinary nominal execution unless the router observes an event."
        ),
    )
    parser.add_argument(
        "--defer-action-mode", choices=("retreat_to_reset", "hold_current"),
        default="retreat_to_reset",
        help="Physical action used while the router selects defer/guard.",
    )
    parser.add_argument(
        "--release-safe-hold-on-confirmed-nominal",
        action="store_true",
        help=(
            "Release the initial defer as soon as the causal router confirms "
            "nominal execution; later observed events can still revise it."
        ),
    )
    parser.add_argument(
        "--terminate-score-on-first-resolution",
        action="store_true",
        help=(
            "End scoring for an environment at its first success or safety "
            "violation, matching natural episodic termination while the "
            "fixed-size vector simulator continues stepping masked actions."
        ),
    )
    parser.add_argument(
        "--router-query-every-step", action="store_true",
        help=(
            "Run the same calibrated causal router on every accumulated "
            "pre-action prefix, reducing event-to-handoff latency."
        ),
    )
    parser.add_argument(
        "--router-query-every-step-after", type=int, default=1,
        help="Keep calibrated snapshot queries before this step.",
    )
    parser.add_argument(
        "--factorized-sweep-dispatch", action="store_true",
        help=(
            "For structured routers, use the checkpoint's group-disjoint "
            "event/direction calibration to dispatch a sweep specialist "
            "before the joint readiness posterior matures."
        ),
    )
    parser.add_argument(
        "--factorized-sweep-dispatch-min-step", type=int, default=1,
        help=(
            "First prefix eligible for factorized sweep dispatch. This may "
            "be frozen to training onset_max + 1 so in-envelope events keep "
            "the checkpoint's standard calibrated handoff."
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
    if args.capture_video:
        kwargs["render_mode"] = "rgb_array"
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
    reverse_paths = [
        Path(path) for path in (
            args.reverse_state_checkpoint
            or [
                "results/learned_recovery_v4/learned_recovery_v4_reverse_state_pilot/"
                "reverse_ejection_state_specialist/seed_9351/reverse_frozen_iter424.pt"
            ]
        )
    ]
    state_specs = {
        "permanent": (Path(args.permanent_state_checkpoint), v4_state.shape[1]),
        "forward": (Path(args.forward_state_checkpoint), v3_state.shape[1]),
    }
    if args.nominal_state_checkpoint:
        state_specs["nominal"] = (
            Path(args.nominal_state_checkpoint), v4_state.shape[1],
        )
    state_agents = {}
    for name, (path, width) in state_specs.items():
        agent = StateAgent(width, action_dim).cuda(); agent.load_state_dict(torch.load(path, map_location=device, weights_only=False)["agent"], strict=True); agent.eval(); state_agents[name] = agent
    reverse_agents = []
    for path in reverse_paths:
        agent = StateAgent(v4_state.shape[1], action_dim).cuda()
        agent.load_state_dict(
            torch.load(path, map_location=device, weights_only=False)["agent"],
            strict=True,
        )
        agent.eval()
        reverse_agents.append(agent)
    threshold = float(router_checkpoint["calibration"]["threshold"])
    class_thresholds = torch.tensor(
        router_checkpoint["calibration"].get(
            "class_thresholds_99_precision", [threshold] * 6,
        ), device=device,
    )
    sweep_dispatch = router_checkpoint.get("calibration", {}).get(
        "factorized_sweep_dispatch_99_precision"
    )
    if args.factorized_sweep_dispatch and sweep_dispatch is None:
        raise ValueError(
            "factorized sweep dispatch requested but checkpoint lacks calibration"
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
            last_action = torch.zeros(
                (args.num_envs,) + env.single_action_space.shape, device=device,
            )
            captured_frames, captured_options = [], []
            captured_resolution = None
            captured_goals = None
            if args.capture_video:
                captured_frames.append(_render_frame(env))
            for step in range(1, args.steps + 1):
                if args.fixed_option is None and step <= router_horizon:
                    feature, positions = extract_features(
                        obs, initial_positions, previous_positions, initial_tcp,
                        initial_qpos, step, router_horizon,
                    )
                    previous_positions = positions
                    history.append(feature)
                if args.fixed_option is None and (
                    step in SNAPSHOTS
                    or (
                        args.router_query_every_step
                        and step >= args.router_query_every_step_after
                    )
                ):
                    output, probability = router_output(
                        router, history, current_centered_geometry_dim,
                    )
                    confidence, proposed = probability.max(1)
                    proposed = torch.where(
                        confidence >= class_thresholds[proposed], proposed,
                        torch.full_like(proposed, 5),
                    )
                    if args.factorized_sweep_dispatch and hasattr(
                        output, "event_logits"
                    ):
                        event_sweep = output.event_logits.softmax(-1)[:, 1]
                        direction_confidence, direction = (
                            output.direction_logits.softmax(-1).max(1)
                        )
                        dispatch = (
                            (step >= args.factorized_sweep_dispatch_min_step)
                            & (event_sweep >= sweep_dispatch["event_threshold"])
                            & (
                                direction_confidence
                                >= sweep_dispatch["direction_threshold"]
                            )
                        )
                        proposed = torch.where(dispatch, direction + 1, proposed)
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
                    & (step >= args.safe_hold_start_step)
                    & (step <= args.safe_hold_until_step)
                    & ~(
                        args.release_safe_hold_on_confirmed_nominal
                        & nominal_confirmed
                    ),
                    torch.full_like(selected_option, 5),
                    selected_option,
                )
                if args.capture_video:
                    captured_options.append(
                        int(effective_option[args.capture_env_index])
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
                reverse_stack = torch.stack([
                    agent.get_action(v4_state, deterministic=True).clamp(-1, 1)
                    for agent in reverse_agents
                ])
                reverse_action = (
                    reverse_stack.median(0).values
                    if args.reverse_ensemble_reduction == "median"
                    else reverse_stack.mean(0)
                )
                defer_action = (
                    torch.zeros(
                        (args.num_envs,) + env.single_action_space.shape,
                        device=device,
                    )
                    if args.defer_action_mode == "hold_current"
                    else retreat_action(
                        obs, initial_qpos, env.single_action_space.shape,
                    )
                )
                if args.defer_action_mode == "hold_current":
                    # Delta-position arm commands hold at zero, while the
                    # gripper requires its previous signed target to avoid
                    # releasing a partially grasped object during the guard.
                    defer_action[:, -1] = last_action[:, -1]
                actions = (
                    nominal_action,
                    state_agents["forward"].get_action(v3_state, deterministic=True).clamp(-1, 1),
                    reverse_action,
                    state_agents["permanent"].get_action(v4_state, deterministic=True).clamp(-1, 1),
                    temporary_action,
                    defer_action,
                )
                stacked = torch.stack(actions, dim=1)
                action = stacked[torch.arange(args.num_envs, device=device), effective_option]
                if args.terminate_score_on_first_resolution:
                    resolved = success | violation
                    action = torch.where(resolved[:, None], torch.zeros_like(action), action)
                last_action = action.clone()
                option_histogram += torch.bincount(effective_option, minlength=6)
                obs, _, _, _, info = env.step(action)
                if args.capture_video:
                    captured_frames.append(_render_frame(env))
                    if captured_resolution is None:
                        index = args.capture_env_index
                        if bool(info["success"][index]) or bool(
                            info["constraint_violated"][index]
                        ):
                            captured_resolution = step
                            # Whether the episode was resolved by completing
                            # the goals or by one becoming unavailable is the
                            # difference between recovery and abandonment.
                            captured_goals = {
                                "goals_completed": float(
                                    info["goals_completed"][index]
                                ),
                                "goals_unavailable": float(
                                    info["goals_unavailable"][index]
                                ),
                            }
                if args.terminate_score_on_first_resolution:
                    active = ~(success | violation)
                    success |= info["success"].bool() & active
                    violation |= info["constraint_violated"].bool() & active
                else:
                    success |= info["success"].bool()
                    violation |= info["constraint_violated"].bool()
            if args.capture_video:
                _write_capture(
                    args, condition, captured_frames, captured_options,
                    captured_resolution, captured_goals, router_checkpoint,
                    bool(success[args.capture_env_index]),
                    bool(violation[args.capture_env_index]),
                )
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
        "safe_hold_start_step": args.safe_hold_start_step,
        "defer_action_mode": args.defer_action_mode,
        "release_safe_hold_on_confirmed_nominal": (
            args.release_safe_hold_on_confirmed_nominal
        ),
        "terminate_score_on_first_resolution": (
            args.terminate_score_on_first_resolution
        ),
        "router_query_every_step": args.router_query_every_step,
        "router_query_every_step_after": args.router_query_every_step_after,
        "factorized_sweep_dispatch": args.factorized_sweep_dispatch,
        "factorized_sweep_dispatch_min_step": (
            args.factorized_sweep_dispatch_min_step
        ),
        "factorized_sweep_dispatch_calibration": sweep_dispatch,
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
        "reverse_ensemble_reduction": args.reverse_ensemble_reduction,
        "reverse_state_checkpoints": [str(path) for path in reverse_paths],
        "reverse_state_checkpoint_sha256": [
            hashlib.sha256(path.read_bytes()).hexdigest() for path in reverse_paths
        ],
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
