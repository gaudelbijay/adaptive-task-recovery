#!/usr/bin/env python3
"""Run the frozen state-policy evaluator under locked physical removal.

The base evaluator remains byte-identical for running cohorts. This wrapper
changes only the evaluation intervention, requires recognized physical goal
unavailability in every episode, and writes a distinct result artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import torch

import evaluate_manipulation_ppo as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict-config", required=True)
    parser.add_argument("--output", default="results/learned_recovery")
    parser.add_argument("--checkpoint-output")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()

    strict_path = Path(args.strict_config)
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    episodes = int(args.episodes or strict["episodes_per_training_seed"])
    num_envs = int(args.num_envs or strict["num_envs"])
    if episodes % num_envs:
        raise ValueError("strict episodes must be divisible by num-envs")
    overrides = dict(strict["intervention_overrides"])
    overrides["onset_step_range"] = tuple(overrides["onset_step_range"])
    training_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = base._select_task(training_config, args.task_index)
    method = task.get("method", task["env_id"])
    checkpoint_root = Path(args.checkpoint_output or args.output)
    checkpoint_path = (
        checkpoint_root / training_config["name"] / method
        / f"seed_{int(task['seed'])}" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")
    registration_module = (
        importlib.import_module(task["registration_module"])
        if task.get("registration_module") else None
    )
    checkpoint_file_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    training_task_sha256 = hashlib.sha256(
        json.dumps(task, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    original_env_kwargs = base._environment_kwargs
    original_atomic_json = base._atomic_json

    def strict_env_kwargs(task, evaluation=False):
        kwargs = original_env_kwargs(task, evaluation=evaluation)
        if evaluation:
            kwargs.update(overrides)
        return kwargs

    def strict_atomic_json(payload, path):
        records = payload.get("episode_records", [])
        if len(records) != episodes:
            raise RuntimeError("strict evaluator produced the wrong episode count")
        required = set(strict["required_episode_fields"])
        for index, record in enumerate(records):
            missing = required - record.keys()
            if missing:
                raise RuntimeError(
                    f"strict episode {index} lacks fields: {sorted(missing)}"
                )
        actual = sum(record["goals_unavailable"] >= 0.5 for record in records)
        if strict["require_actual_goal_unavailable_every_episode"] and actual != episodes:
            raise RuntimeError(
                "strict intervention failed to make a goal unavailable in "
                f"{episodes - actual}/{episodes} episodes"
            )
        safe_successes = sum(
            base._metric_success(record) >= 0.5
            and record["constraint_violated"] < 0.5
            for record in records
        )
        payload["protocol"] = strict["protocol"]
        payload["condition"] = "strict_intervention"
        payload["safe_successes"] = safe_successes
        payload["safe_success_rate"] = safe_successes / episodes
        payload["training_source_sha256"] = checkpoint.get("source_sha256")
        payload["checkpoint_file_sha256"] = checkpoint_file_sha256
        payload["training_task_sha256"] = training_task_sha256
        payload["strict_removal"] = {
            "actual_unavailable_episodes": actual,
            "intervention_overrides": overrides,
            "require_actual_goal_unavailable_every_episode": True,
            "claim_boundary": strict["claim_boundary"],
            "protocol_calibration": strict.get("protocol_calibration"),
        }
        payload["evaluation_source_sha256"] = {
            "evaluator": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            "strict_evaluator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "strict_config": hashlib.sha256(strict_path.read_bytes()).hexdigest(),
            "evaluation_seed": hashlib.sha256(
                (Path(__file__).parent / "evaluation_seed.py").read_bytes()
            ).hexdigest(),
            "environment_registration": (
                hashlib.sha256(Path(registration_module.__file__).read_bytes()).hexdigest()
                if registration_module is not None and registration_module.__file__
                else None
            ),
        }
        original_atomic_json(
            payload, path.with_name("heldout_eval_strict_intervention.json")
        )

    base._environment_kwargs = strict_env_kwargs
    base._atomic_json = strict_atomic_json
    argv = [
        str(Path(base.__file__)),
        "--config", args.config,
        "--output", args.output,
        "--task-index", str(args.task_index),
        "--episodes", str(episodes),
        "--num-envs", str(num_envs),
        "--seed-base", str(strict["seed_base"]),
        "--condition", "intervention",
    ]
    if args.checkpoint_output:
        argv.extend(["--checkpoint-output", args.checkpoint_output])
    sys.argv = argv
    base.main()


if __name__ == "__main__":
    main()
