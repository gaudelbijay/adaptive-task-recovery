#!/usr/bin/env python3
"""Matched closed-loop evaluation for the external PegInsertion recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401
from atr.policies.causal_option_router import (
    CausalOptionRouter, StaticOptionRouter, UnstructuredOptionGRU,
    current_centered_sequence,
)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_manipulation_ppo import Agent


CONDITIONS = (
    "nominal", "positive_lateral_peg_ejection", "permanent_hole_block",
    "temporary_hole_block", "negative_lateral_peg_ejection",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_geometry(raw: torch.Tensor) -> torch.Tensor:
    pose = raw.reshape(raw.shape[0], 4, 7)
    peg, hole, blocker, tcp = (pose[:, index, :3] for index in range(4))
    return torch.cat((peg - hole, blocker - hole, tcp - peg, tcp - hole), dim=1)


def load_agent(path: Path, observation_dim: int, action_dim: int, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    agent = Agent(observation_dim, action_dim).to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    agent.eval()
    return agent, checkpoint


def load_router(path: Path, metadata_path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["feature_metadata_sha256"] != sha256(metadata_path):
        raise ValueError("router feature metadata hash mismatch")
    name = checkpoint["model"]
    if name == "causal_gru":
        model = CausalOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    elif name == "static_mlp":
        model = StaticOptionRouter(checkpoint["input_dim"], checkpoint["hidden_dim"])
    elif name == "unstructured_gru":
        model = UnstructuredOptionGRU(checkpoint["input_dim"], checkpoint["hidden_dim"], 2)
    else:
        raise ValueError(f"unknown router model: {name}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def learned_option(model, checkpoint, history, geometry_dim: int):
    sequence = torch.stack(history, dim=1)
    length = torch.full(
        (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
        device=sequence.device,
    )
    sequence = current_centered_sequence(sequence, length, geometry_dim)
    output = model(sequence, length)
    logp = output.option_log_probability if hasattr(output, "option_log_probability") else output
    probability = logp.exp()
    confidence, option = probability.max(1)
    thresholds = torch.as_tensor(
        checkpoint["calibration"]["class_thresholds_99_precision"],
        device=sequence.device,
    )
    accepted = confidence >= thresholds[option]
    return torch.where(accepted, option, torch.full_like(option, 5)), confidence


def heuristic_option(geometry, initial_geometry, previous_geometry, blocker_seen):
    peg_tcp_delta_y = geometry[:, 7] - previous_geometry[:, 7]
    initial_blocker_distance = torch.linalg.vector_norm(initial_geometry[:, 3:6], dim=1)
    blocker_distance = torch.linalg.vector_norm(geometry[:, 3:6], dim=1)
    # The unused dynamic blocker has small servo/contact settling even in
    # nominal episodes. Only substantial *inward* travel toward the hole is
    # evidence of obstruction; arbitrary displacement is not.
    blocker_progress = initial_blocker_distance - blocker_distance
    blocker_now = blocker_progress > 0.035
    blocker_cleared = blocker_seen & (blocker_progress < 0.015)
    blocker_seen |= blocker_now
    option = torch.zeros(len(geometry), dtype=torch.long, device=geometry.device)
    option[peg_tcp_delta_y < -0.03] = 1
    option[peg_tcp_delta_y > 0.03] = 2
    option[blocker_now] = 5
    option[blocker_cleared] = 4
    return option


def oracle_option(info):
    mechanism = info["critic_intervention_mechanism"].long()
    option = torch.zeros_like(mechanism)
    finished = info["intervention_finished"].bool()
    option[(mechanism == 0) & finished] = 1
    option[(mechanism == 3) & finished] = 2
    option[(mechanism == 1) & info["critic_physical_unavailable"].bool()] = 3
    option[(mechanism == 2) & info["temporary_cleared"].bool()] = 4
    waiting = (mechanism >= 0) & ~finished
    option[waiting] = 5
    return option


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--method", choices=("causal_gru", "static_mlp", "unstructured_gru", "heuristic", "oracle"),
        required=True,
    )
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--router-metadata", type=Path)
    parser.add_argument("--nominal-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--forward-checkpoint", action="append", type=Path)
    parser.add_argument("--reverse-checkpoint", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=425_000_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--router-horizon", type=int, default=96)
    parser.add_argument("--abstention-steps", type=int, default=8)
    args = parser.parse_args()
    if args.episodes != args.num_envs:
        raise ValueError("one task must evaluate exactly one vector batch")
    if not 0 <= args.task_index < 15:
        raise ValueError("task-index must be in [0, 14]")
    evaluation_seed_index, condition_index = divmod(args.task_index, len(CONDITIONS))
    condition = CONDITIONS[condition_index]
    device = torch.device("cuda")
    nominal_tasks = [
        torch.load(path, map_location="cpu", weights_only=False)["task"]
        for path in args.nominal_checkpoint
    ]
    blocker_observation_flags = {
        bool(task.get("env_kwargs", {}).get("include_blocker_state_observation", True))
        for task in nominal_tasks
    }
    if len(blocker_observation_flags) != 1:
        raise ValueError("nominal checkpoints disagree on blocker observation contract")
    include_blocker_state_observation = next(iter(blocker_observation_flags))
    env = gym.make(
        "PegInsertionRecovery-v1", num_envs=args.num_envs, reconfiguration_freq=1,
        obs_mode="state", render_mode=None, sim_backend="physx_cuda",
        control_mode="pd_joint_delta_pos", reward_mode="normalized_dense",
        intervention_probability=0.0 if condition == "nominal" else 1.0,
        intervention_types=(
            ("positive_lateral_peg_ejection",) if condition == "nominal" else (condition,)
        ),
        onset_step_range=(18, 42), max_episode_steps=args.steps,
        include_blocker_state_observation=include_blocker_state_observation,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=False)
    seed = args.seed_base + evaluation_seed_index * 1_000_000 + condition_index * 10_000
    observation, info = env.reset(seed=seed)
    observation_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)
    nominal = [load_agent(path, observation_dim, action_dim, device)[0] for path in args.nominal_checkpoint]
    forward = [load_agent(path, observation_dim, action_dim, device)[0] for path in (args.forward_checkpoint or args.nominal_checkpoint)]
    reverse = [load_agent(path, observation_dim, action_dim, device)[0] for path in (args.reverse_checkpoint or args.nominal_checkpoint)]
    router = router_checkpoint = None
    geometry_dim = 12
    router_seed = None
    if args.method in ("causal_gru", "static_mlp", "unstructured_gru"):
        if args.router_checkpoint is None or args.router_metadata is None:
            raise ValueError("learned methods require router checkpoint and metadata")
        router, router_checkpoint = load_router(args.router_checkpoint, args.router_metadata, device)
        geometry_dim = int(router_checkpoint["current_centered_geometry_dim"])
        router_seed = int(router_checkpoint["seed"])

    initial_geometry = relative_geometry(info["router_task_geometry"])
    previous_geometry = initial_geometry.clone()
    history = []
    blocker_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    active = torch.ones_like(blocker_seen)
    success_once = torch.zeros_like(active)
    violation_once = torch.zeros_like(active)
    safe_abstention = torch.zeros_like(active)
    abstention_run = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    decision_step = torch.full_like(abstention_run, -1)
    recovery_step = torch.full_like(abstention_run, -1)
    final_option = torch.zeros(args.num_envs, dtype=torch.long, device=device)
    confidence = torch.zeros(args.num_envs, device=device)
    option_counts = torch.zeros(6, dtype=torch.long, device=device)
    maximum_blocker_inward_progress = torch.zeros(args.num_envs, device=device)
    maximum_peg_tcp_y_step = torch.zeros(args.num_envs, device=device)
    try:
        for step in range(1, args.steps + 1):
            geometry = relative_geometry(info["router_task_geometry"])
            blocker_progress = (
                torch.linalg.vector_norm(initial_geometry[:, 3:6], dim=1)
                - torch.linalg.vector_norm(geometry[:, 3:6], dim=1)
            )
            maximum_blocker_inward_progress = torch.maximum(
                maximum_blocker_inward_progress, blocker_progress,
            )
            maximum_peg_tcp_y_step = torch.maximum(
                maximum_peg_tcp_y_step,
                (geometry[:, 7] - previous_geometry[:, 7]).abs(),
            )
            feature = torch.cat((
                geometry,
                torch.full((args.num_envs, 1), step / args.router_horizon, device=device),
            ), dim=1)
            if len(history) < args.router_horizon:
                history.append(feature)
            if args.method in ("causal_gru", "static_mlp", "unstructured_gru"):
                final_option, confidence = learned_option(
                    router, router_checkpoint, history, geometry_dim,
                )
            elif args.method == "heuristic":
                final_option = heuristic_option(
                    geometry, initial_geometry, previous_geometry, blocker_seen,
                )
                confidence.fill_(1.0)
            else:
                final_option = oracle_option(info)
                confidence.fill_(1.0)
            option_counts += torch.bincount(final_option, minlength=6)
            previous_geometry = geometry.clone()
            newly_decided = (decision_step < 0) & (final_option != 0)
            decision_step[newly_decided] = step

            nominal_action = torch.stack([agent.get_action(observation, True) for agent in nominal]).mean(0)
            forward_action = torch.stack([agent.get_action(observation, True) for agent in forward]).mean(0)
            reverse_action = torch.stack([agent.get_action(observation, True) for agent in reverse]).mean(0)
            action = nominal_action.clone()
            action[final_option == 1] = forward_action[final_option == 1]
            action[final_option == 2] = reverse_action[final_option == 2]
            abstaining = (final_option == 3) | (final_option == 5)
            action[abstaining] = 0
            action[~active] = 0
            action = torch.clamp(action, action_low, action_high)
            observation, _, _, _, info = env.step(action)

            violation = info["constraint_violated"].bool() & active
            violation_once |= violation
            available_success = info["success"].bool()
            if condition != "nominal":
                available_success &= info["intervention_finished"].bool()
            success = available_success & active & ~violation_once
            success_once |= success
            recovery_step[(recovery_step < 0) & success] = step
            unavailable = info["critic_physical_unavailable"].bool() & active
            abstention_run = torch.where(
                unavailable & abstaining,
                abstention_run + 1,
                torch.zeros_like(abstention_run),
            )
            abstained = unavailable & (abstention_run >= args.abstention_steps) & ~violation_once
            safe_abstention |= abstained
            resolved = violation | success | abstained
            active &= ~resolved
            if not bool(active.any()):
                break
    finally:
        env.close()

    safe_outcome = safe_abstention if condition == "permanent_hole_block" else success_once
    result = {
        "schema_version": 1,
        "environment": "PegInsertionRecovery-v1",
        "method": args.method,
        "router_seed": router_seed,
        "condition": condition,
        "evaluation_seed_index": evaluation_seed_index,
        "seed_base": args.seed_base,
        "episode_seed": seed,
        "episodes": args.episodes,
        "successes": int(success_once.sum()),
        "safe_abstentions": int(safe_abstention.sum()),
        "safe_successes": int(safe_outcome.sum()),
        "violations": int(violation_once.sum()),
        "safe_success_rate": float(safe_outcome.float().mean()),
        "violation_rate": float(violation_once.float().mean()),
        "episode_ids": [seed + index for index in range(args.episodes)],
        "episode_safe_outcome": safe_outcome.cpu().tolist(),
        "episode_native_success": success_once.cpu().tolist(),
        "episode_safe_abstention": safe_abstention.cpu().tolist(),
        "episode_violation": violation_once.cpu().tolist(),
        "mean_decision_step": float(decision_step[decision_step >= 0].float().mean()) if bool((decision_step >= 0).any()) else None,
        "mean_recovery_step": float(recovery_step[recovery_step >= 0].float().mean()) if bool((recovery_step >= 0).any()) else None,
        "final_option_histogram": torch.bincount(final_option, minlength=6).cpu().tolist(),
        "option_step_histogram": option_counts.cpu().tolist(),
        "mean_final_confidence": float(confidence.mean()),
        "observable_diagnostics": {
            "maximum_blocker_inward_progress_mean": float(maximum_blocker_inward_progress.mean()),
            "maximum_blocker_inward_progress_q95": float(torch.quantile(maximum_blocker_inward_progress, 0.95)),
            "maximum_peg_tcp_y_step_mean": float(maximum_peg_tcp_y_step.mean()),
            "maximum_peg_tcp_y_step_q95": float(torch.quantile(maximum_peg_tcp_y_step, 0.95)),
        },
        "router_checkpoint": str(args.router_checkpoint) if args.router_checkpoint else None,
        "router_checkpoint_sha256": sha256(args.router_checkpoint) if args.router_checkpoint else None,
        "router_metadata_sha256": sha256(args.router_metadata) if args.router_metadata else None,
        "nominal_checkpoint_sha256": [sha256(path) for path in args.nominal_checkpoint],
        "forward_checkpoint_sha256": [sha256(path) for path in (args.forward_checkpoint or args.nominal_checkpoint)],
        "reverse_checkpoint_sha256": [sha256(path) for path in (args.reverse_checkpoint or args.nominal_checkpoint)],
        "include_blocker_state_observation": include_blocker_state_observation,
        "forbidden_runtime_inputs": [
            "intervention kind", "intervention target", "future observation",
            "oracle feasibility", "native success flag",
        ] if args.method != "oracle" else [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"eval_{evaluation_seed_index}_{condition}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
