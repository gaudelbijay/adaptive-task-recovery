#!/usr/bin/env python3
"""Apply the frozen three-seed competence gate to PegInsertion PPO audits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/external_peg_nominal_ppo_v1.json"),
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/a_plus_audit/external_peg_nominal_ppo_v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/a_plus_audit/external_peg_nominal_ppo_v1.json"),
    )
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    records = [
        json.loads(path.read_text())
        for path in sorted(args.input_dir.glob("seed_*.json"))
    ]
    expected = set(int(seed) for seed in config["seeds"])
    observed = {int(record["training_seed"]) for record in records}
    if observed != expected or len(records) != len(expected):
        raise RuntimeError(f"expected training seeds {sorted(expected)}, found {sorted(observed)}")
    evaluation_seeds = {int(record["evaluation_seed_base"]) for record in records}
    if len(evaluation_seeds) != 1:
        raise RuntimeError(f"mixed evaluation seed bases: {sorted(evaluation_seeds)}")

    criteria = config["competence_gate"]
    episodes = sum(int(record["episodes"]) for record in records)
    successes = sum(int(record["safe_successes"]) for record in records)
    violations = sum(int(record["violations"]) for record in records)
    per_seed = {
        str(record["training_seed"]): {
            "safe_success_rate": record["safe_success_rate"],
            "violation_rate": record["violation_rate"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "checkpoint_global_step": record["checkpoint_global_step"],
        }
        for record in records
    }
    checks = {
        "three_seed_mean_safe_success": (
            successes / episodes
            >= criteria["minimum_three_seed_mean_safe_success"]
        ),
        "minimum_per_seed_safe_success": (
            min(record["safe_success_rate"] for record in records)
            >= criteria["minimum_per_seed_safe_success"]
        ),
        "constraint_violation_rate": (
            violations / episodes
            <= criteria["maximum_constraint_violation_rate"]
        ),
    }
    result = {
        "schema_version": 1,
        "audit": "official_ppo_nominal_competence",
        "evaluation_seed_base": next(iter(evaluation_seeds)),
        "episodes": episodes,
        "safe_successes": successes,
        "safe_success_rate": successes / episodes,
        "violations": violations,
        "violation_rate": violations / episodes,
        "per_training_seed": per_seed,
        "checks": checks,
        "competence_gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_reject and not result["competence_gate_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
