#!/usr/bin/env python3
"""Run the frozen visual evaluator under guaranteed early physical removal.

This wrapper deliberately leaves ``evaluate_visual_recovery_ppo.py`` byte-
identical for already-running factorial cohorts. It overrides only evaluation
environment intervention timing, requires actual recognized unavailability in
every episode, and writes a distinct result file and protocol marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import evaluate_visual_recovery_ppo as base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict-config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--num-envs", type=int)
    args = parser.parse_args()

    strict = json.loads(Path(args.strict_config).read_text(encoding="utf-8"))
    episodes = int(args.episodes or strict["episodes_per_training_seed"])
    num_envs = int(args.num_envs or strict["num_envs"])
    if episodes % num_envs:
        raise ValueError("strict episodes must be divisible by num-envs")
    overrides = dict(strict["intervention_overrides"])
    overrides["onset_step_range"] = tuple(overrides["onset_step_range"])

    original_env_kwargs = base.env_kwargs
    original_atomic_json = base.atomic_json

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
                f"strict intervention failed to make a goal unavailable in "
                f"{episodes - actual}/{episodes} episodes"
            )
        payload["protocol"] = strict["protocol"]
        payload["condition"] = "strict_intervention"
        payload["strict_removal"] = {
            "actual_unavailable_episodes": actual,
            "intervention_overrides": overrides,
            "require_actual_goal_unavailable_every_episode": True,
            "claim_boundary": strict["claim_boundary"],
            "protocol_calibration": strict.get("protocol_calibration"),
        }
        payload["evaluation_source_sha256"]["strict_evaluator"] = (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        payload["evaluation_source_sha256"]["strict_config"] = (
            hashlib.sha256(Path(args.strict_config).read_bytes()).hexdigest()
        )
        target = path.with_name("heldout_eval_strict_intervention.json")
        original_atomic_json(payload, target)

    base.env_kwargs = strict_env_kwargs
    base.atomic_json = strict_atomic_json
    sys.argv = [
        str(Path(base.__file__)),
        "--config", args.config,
        "--output", args.output,
        "--task-index", str(args.task_index),
        "--episodes", str(episodes),
        "--num-envs", str(num_envs),
        "--seed-base", str(strict["seed_base"]),
        "--condition", "intervention",
    ]
    base.main()


if __name__ == "__main__":
    main()
