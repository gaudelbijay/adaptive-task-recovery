#!/usr/bin/env python3
"""Capture declared recovery branches from a frozen restricted-input visual policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_visual_recovery_ppo import (
    VisualAgent,
    env_kwargs,
    extract_observation,
    observation_contract,
    privileged_aux_dim,
    select_task,
    visual_progress_target,
)
from check_visual_competence_gate import check_visualization_gate


def _frame(env) -> np.ndarray:
    image = env.render()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8, copy=False)


def _scalar(info: dict, key: str, default: bool = False) -> bool:
    value = info.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--strict-config",
        help="apply the locked strict-removal intervention to non-nominal captures",
    )
    parser.add_argument("--results", default="results/visual_recovery_ppo")
    parser.add_argument("--output", default="results/visual_recovery_ppo/videos")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--branch", choices=("first_goal_removed", "second_goal_removed", "nominal"),
        required=True,
    )
    parser.add_argument("--seed-base", type=int, default=92000000)
    parser.add_argument("--max-attempts", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--aggregate")
    parser.add_argument(
        "--selection-artifact",
        help="frozen integrated-policy selection artifact authorizing capture",
    )
    parser.add_argument("--expected-selection")
    parser.add_argument("--minimum-raw-success", type=float, default=425 / 768)
    parser.add_argument("--minimum-safe-success", type=float, default=424 / 768)
    parser.add_argument("--maximum-violation", type=float, default=12 / 768)
    parser.add_argument("--minimum-nominal-success", type=float, default=0.70)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("visual policy capture requires a CUDA GPU")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = select_task(config, args.task_index)
    registration_module = None
    if task.get("registration_module"):
        registration_module = importlib.import_module(task["registration_module"])
    training_seed = int(task["seed"])
    method = task["method"]
    run_dir = Path(args.results) / config["name"] / method / f"seed_{training_seed}"
    if args.selection_artifact:
        if args.aggregate:
            raise ValueError("use either an aggregate or a selection artifact")
        if not args.expected_selection:
            raise ValueError("integrated capture requires --expected-selection")
        selection_path = Path(args.selection_artifact)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("protocol") != "predeclared integrated visual-policy selection":
            raise ValueError("integrated capture selection has the wrong protocol")
        if selection.get("selected") != args.expected_selection:
            raise RuntimeError("integrated capture candidate was not selected")
        candidates = [
            item for item in selection.get("candidates", [])
            if item.get("label") == args.expected_selection
        ]
        if len(candidates) != 1:
            raise ValueError("integrated capture selection has the wrong candidate count")
        candidate = candidates[0]
        if (
            not candidate.get("eligible") or not candidate.get("checks")
            or not all(candidate["checks"].values())
        ):
            raise RuntimeError("integrated capture candidate lacks complete eligibility")
        if candidate.get("method") != method:
            raise ValueError("integrated capture task does not match selected method")
        visualization_gate = {
            "passed": True,
            "protocol": selection["protocol"],
            "selected": selection["selected"],
            "candidate": candidate,
            "thresholds": selection.get("thresholds"),
            "selection_artifact": str(selection_path),
            "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
            "source_sha256": selection.get("source_sha256"),
        }
        aggregate_path = None
    else:
        aggregate_path = Path(args.aggregate) if args.aggregate else (
            Path(args.results) / config["name"] / "aggregate.json"
        )
        if not aggregate_path.exists():
            raise FileNotFoundError(f"validated aggregate unavailable: {aggregate_path}")
        visualization_gate = check_visualization_gate(
            json.loads(aggregate_path.read_text(encoding="utf-8")), method,
            args.minimum_raw_success, args.minimum_safe_success,
            args.maximum_violation, args.minimum_nominal_success,
        )
    if not visualization_gate["passed"]:
        raise RuntimeError(
            f"policy is ineligible for README visualization: {visualization_gate}"
        )
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")
    if checkpoint.get("observation_contract") != observation_contract(task):
        raise ValueError("checkpoint lacks the declared restricted visual contract")

    strict = None
    if args.strict_config:
        strict_path = Path(args.strict_config)
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
    kwargs = env_kwargs(task, evaluation=True)
    kwargs["render_mode"] = "rgb_array"
    kwargs["intervention_probability"] = 0.0 if args.branch == "nominal" else 1.0
    if strict is not None and args.branch != "nominal":
        kwargs.update(strict["intervention_overrides"])
        kwargs["onset_step_range"] = tuple(kwargs["onset_step_range"])
    env = gym.make(task["env_id"], num_envs=1, reconfiguration_freq=1, **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    observation, _ = env.reset(seed=args.seed_base)
    rgb, proprio, critic = extract_observation(
        observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
        task.get("actor_goal_progress", False),
    )
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = VisualAgent(
        task["image_size"], proprio.shape[1], critic.shape[1], action_dim,
        task["asymmetric_critic"], 0, privileged_aux_dim(task),
        task.get("actor_learned_goal_progress", False),
    ).cuda()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    def rollout(episode_seed: int, capture: bool = False):
        observation, _ = env.reset(seed=episode_seed)
        frames = [_frame(env)] if capture else []
        branch_matches = args.branch == "nominal"
        actual_removal_once = False
        success_once = False
        violation_once = False
        progress_correct = 0
        progress_total = 0
        steps = 0
        for steps in range(1, int(task["num_eval_steps"]) + 1):
            rgb, proprio, _ = extract_observation(
                observation, task["asymmetric_critic"], task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            if agent.goal_progress_predictor is not None:
                prediction = (
                    torch.sigmoid(agent.goal_progress_predictor(agent.encode(rgb))) >= 0.5
                )
                matches = prediction == visual_progress_target(observation).bool()
                progress_correct += int(matches.sum())
                progress_total += int(matches.numel())
            action = agent.get_action(rgb, proprio, deterministic=True)
            observation, _, _, truncated, info = env.step(action)
            if steps == 1 and args.branch != "nominal":
                first_removed = _scalar(info, "first_goal_removed")
                branch_matches = first_removed == (args.branch == "first_goal_removed")
            success_once |= _scalar(info, "success")
            actual_removal_once |= _scalar(info, "goals_unavailable")
            violation_once |= _scalar(info, "constraint_violated")
            if capture:
                frames.append(_frame(env))
            if success_once or violation_once or _scalar({"done": truncated}, "done"):
                break
        accuracy = progress_correct / progress_total if progress_total else None
        return (
            branch_matches, actual_removal_once, success_once, violation_once,
            steps, frames, accuracy,
        )

    selected = None
    with torch.no_grad():
        for attempt in range(args.max_attempts):
            episode_seed = args.seed_base + attempt
            (
                branch_matches, actual_removal_once, success_once,
                violation_once, steps, _, _,
            ) = rollout(episode_seed)
            removal_requirement = args.branch == "nominal" or actual_removal_once
            if branch_matches and removal_requirement and success_once and not violation_once:
                selected = (episode_seed, steps)
                break
    if selected is None:
        env.close()
        raise RuntimeError(
            f"no safe successful {args.branch} episode found in "
            f"{args.max_attempts} declared seeds beginning at {args.seed_base}"
        )

    episode_seed, search_steps = selected
    with torch.no_grad():
        (
            branch_matches, actual_removal_once, success_once, violation_once,
            steps, frames, progress_accuracy,
        ) = rollout(episode_seed, capture=True)
    env.close()
    removal_requirement = args.branch == "nominal" or actual_removal_once
    if (
        not branch_matches or not removal_requirement or not success_once
        or violation_once or steps != search_steps
    ):
        raise RuntimeError("deterministic rendered replay did not match qualifying rollout")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_{training_seed}_{args.branch}"
    video_path = output_dir / f"{stem}.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=2)
    metadata = {
        "schema_version": 1,
        "protocol": (
            "first safe success with actual physical goal unavailability in a "
            "declared sequential seed range"
            if args.branch != "nominal"
            else "first safe nominal success in a declared sequential seed range"
        ),
        "benchmark_semantics": (
            "event_reward_intervention_target_only_v3"
            if task["env_id"] == "LearnedRecovery-v3"
            else "intervention_target_only_v2"
        ),
        "env_id": task["env_id"], "method": method, "branch": args.branch,
        "training_seed": training_seed, "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "observation_contract": checkpoint["observation_contract"],
        "seed_base": args.seed_base, "max_attempts": args.max_attempts,
        "episode_seed": episode_seed, "steps": steps, "frames": len(frames),
        "fps": args.fps, "safe_success": True, "teleport_calls": 0,
        "actual_goal_unavailable": actual_removal_once,
        "visual_resolution_bit_accuracy": progress_accuracy,
        "aggregate": str(aggregate_path) if aggregate_path is not None else None,
        "visualization_gate": visualization_gate,
        "strict_removal_config": (
            {
                "path": args.strict_config,
                "sha256": hashlib.sha256(Path(args.strict_config).read_bytes()).hexdigest(),
                "intervention_overrides": strict["intervention_overrides"],
            }
            if strict is not None and args.branch != "nominal" else None
        ),
        "training_source_sha256": checkpoint.get("source_sha256"),
        "capture_source_sha256": {
            "capture": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment_registration": (
                hashlib.sha256(Path(registration_module.__file__).read_bytes()).hexdigest()
                if registration_module is not None and registration_module.__file__
                else None
            ),
        },
    }
    _atomic_json(metadata, output_dir / f"{stem}.json")
    print(json.dumps({**metadata, "video": str(video_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
