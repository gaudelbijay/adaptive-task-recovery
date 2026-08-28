#!/usr/bin/env python3
"""Aggregate a multi-seed NE-Dreamer pilot from metrics and completion markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()

    root = Path(args.root)
    per_seed = []
    for seed_dir in sorted(root.glob("seed_*")):
        training_marker = seed_dir / "TRAINING_COMPLETE.json"
        evaluation_marker = seed_dir / f"EVALUATION_{args.eval_episodes}_COMPLETE.json"
        if not training_marker.exists() or not evaluation_marker.exists():
            raise RuntimeError(f"incomplete seed directory: {seed_dir}")
        records = _records(seed_dir / "metrics.jsonl")
        evaluations = [record for record in records if "episode/eval_success_rate" in record]
        train = [record for record in records if "train/loss/ne_dreamer" in record]
        final = evaluations[-1]
        finite_fields = [
            value
            for record in train
            for key, value in record.items()
            if key.startswith("train/") and isinstance(value, (int, float))
        ]
        per_seed.append({
            "seed": int(seed_dir.name.removeprefix("seed_")),
            "environment_steps": int(final["step"]),
            "evaluation_episodes": args.eval_episodes,
            "evaluation_successes": int(round(final["episode/eval_success_rate"] * args.eval_episodes)),
            "evaluation_success_rate": final["episode/eval_success_rate"],
            "evaluation_return": final["episode/eval_score"],
            "best_checkpoint_success_rate": max(record["episode/eval_success_rate"] for record in evaluations[:-1]),
            "best_checkpoint_return": max(record["episode/eval_score"] for record in evaluations[:-1]),
            "representation_loss_first": train[0]["train/loss/ne_dreamer"],
            "representation_loss_last": train[-1]["train/loss/ne_dreamer"],
            "dynamics_loss_last": train[-1]["train/loss/dyn"],
            "optimizer_updates": int(train[-1]["train/opt/updates"]),
            "action_min": min(record["train/action_min"] for record in train),
            "action_max": max(record["train/action_max"] for record in train),
            "all_training_metrics_finite": bool(np.isfinite(finite_fields).all()),
        })

    total_episodes = args.eval_episodes * len(per_seed)
    total_successes = sum(item["evaluation_successes"] for item in per_seed)
    if total_successes == 0:
        exact_two_sided_upper_95 = 1.0 - 0.025 ** (1.0 / total_episodes)
    else:
        exact_two_sided_upper_95 = None
    aggregate = {
        "schema_version": 1,
        "algorithm": "NE-Dreamer",
        "observation_protocol": ["rgb_64x64", "joint_position_velocity", "instruction"],
        "privileged_policy_observations": False,
        "teleport_control": False,
        "seeds": per_seed,
        "aggregate": {
            "seed_count": len(per_seed),
            "environment_steps_per_seed": min(item["environment_steps"] for item in per_seed),
            "optimizer_updates_per_seed": min(item["optimizer_updates"] for item in per_seed),
            "evaluation_episodes": total_episodes,
            "evaluation_successes": total_successes,
            "evaluation_success_rate": total_successes / total_episodes,
            "zero_success_exact_two_sided_upper_95": exact_two_sided_upper_95,
            "evaluation_return_mean": float(np.mean([item["evaluation_return"] for item in per_seed])),
            "evaluation_return_std_across_seeds": float(np.std([item["evaluation_return"] for item in per_seed], ddof=1)),
            "all_training_metrics_finite": all(item["all_training_metrics_finite"] for item in per_seed),
            "action_range_observed": [
                min(item["action_min"] for item in per_seed),
                max(item["action_max"] for item in per_seed),
            ],
            "representation_loss_reduction_fraction_mean": float(np.mean([
                1.0 - item["representation_loss_last"] / item["representation_loss_first"]
                for item in per_seed
            ])),
        },
    }
    rendered = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
