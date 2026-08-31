#!/usr/bin/env python3
"""Aggregate process-isolated official PegInsertion competence episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/a_plus_audit"))
    parser.add_argument("--expected-episodes", type=int, default=32)
    parser.add_argument("--minimum-success-rate", type=float, default=0.75)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/a_plus_audit/external_peg_nominal_controller_v1.json"),
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("external_peg_nominal_controller_v1_*.json"))
    rows = []
    for path in paths:
        record = json.loads(path.read_text())
        rows.extend(record["episodes_detail"])
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != args.expected_episodes or len(set(seeds)) != len(rows):
        raise RuntimeError(
            f"expected {args.expected_episodes} unique episodes, found {len(rows)}"
        )
    safe_successes = sum(bool(row["safe_success"]) for row in rows)
    result = {
        "schema_version": 1,
        "audit": "official_nominal_controller_competence",
        "execution": "one isolated native planner process per episode",
        "environment": "PegInsertionRecovery-v1",
        "base_benchmark": "PegInsertionSide-v1",
        "controller": "ManiSkill3 official Panda motion-planning solution",
        "intervention_probability": 0.0,
        "episodes": len(rows),
        "safe_successes": safe_successes,
        "safe_success_rate": safe_successes / len(rows),
        "native_success_rate": sum(bool(row["native_success"]) for row in rows)
        / len(rows),
        "constraint_violation_rate": sum(
            bool(row["constraint_violation"]) for row in rows
        ) / len(rows),
        "planning_failure_rate": sum(bool(row["planning_failed"]) for row in rows)
        / len(rows),
        "minimum_success_rate": args.minimum_success_rate,
        "pass": safe_successes / len(rows) >= args.minimum_success_rate,
        "episodes_detail": sorted(rows, key=lambda row: row["seed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
