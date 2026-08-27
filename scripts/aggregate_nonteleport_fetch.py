#!/usr/bin/env python3
"""Aggregate paired physical-pipeline episodes with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr.evaluation.harness import bootstrap_ci


def metrics(record: dict) -> dict[str, float]:
    per_goal = record["per_goal"]
    recovery = per_goal["place_cracker_box"]
    return {
        "achievable_goal_completion": float(per_goal["place_potted_meat_can"]["achieved"]),
        "goals_achieved": float(record["goals_achieved"]),
        "wasted_steps": float(record["wasted_steps"]),
        "destroyed_goal_wasted_steps": float(
            recovery["steps_used"] if not recovery["achieved"] and not recovery["skipped"] else 0
        ),
        "constraint_violations": float(sum(record["constraint_violations"].values())),
        "visual_feasibility_accuracy": float(recovery["perceived_feasible"] is False),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/nonteleport_fetch/eval")
    parser.add_argument("--output", default="results/nonteleport_fetch/summary.json")
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    root = Path(args.input_dir)
    policies = ("static", "oracle", "visual_learned_guarded")
    records = {
        policy: [json.loads((root / f"{policy}_seed_{seed}.json").read_text()) for seed in range(args.seeds)]
        for policy in policies
    }
    values = {policy: [metrics(row) for row in rows] for policy, rows in records.items()}
    names = tuple(values[policies[0]][0])
    summary = {
        policy: {
            name: dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci(
                [row[name] for row in values[policy]], n_resamples=10000,
            )))
            for name in names
        }
        for policy in policies
    }
    paired = {}
    for comparison, left, right in (
        ("learned_minus_static", "visual_learned_guarded", "static"),
        ("learned_minus_oracle", "visual_learned_guarded", "oracle"),
    ):
        paired[comparison] = {
            name: dict(zip(("mean", "ci_low", "ci_high"), bootstrap_ci([
                values[left][seed][name] - values[right][seed][name]
                for seed in range(args.seeds)
            ], n_resamples=10000)))
            for name in names
        }
    payload = {
        "schema_version": 1,
        "episodes_per_policy": args.seeds,
        "teleport_calls": sum(row["teleport_calls"] for rows in records.values() for row in rows),
        "summary": summary,
        "paired_differences": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
