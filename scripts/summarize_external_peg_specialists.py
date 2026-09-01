#!/usr/bin/env python3
"""Fail-closed three-seed competence gate for external Peg specialists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-mean-safe-success", type=float, default=0.75)
    parser.add_argument("--minimum-seed-safe-success", type=float, default=0.60)
    parser.add_argument("--maximum-violation-rate", type=float, default=0.03)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("*.json"))]
    conditions = (
        "positive_lateral_peg_ejection", "negative_lateral_peg_ejection",
    )
    by_condition = {}
    checks = {}
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        seeds = {record["training_seed"] for record in subset}
        if len(subset) != 3 or len(seeds) != 3:
            raise RuntimeError(f"expected three independent {condition} records")
        episodes = sum(record["episodes"] for record in subset)
        safe = sum(record["safe_successes"] for record in subset)
        violations = sum(record["violations"] for record in subset)
        by_condition[condition] = {
            "episodes": episodes,
            "safe_successes": safe,
            "safe_success_rate": safe / episodes,
            "violation_rate": violations / episodes,
            "per_seed_safe_success": {
                str(record["training_seed"]): record["safe_success_rate"]
                for record in subset
            },
        }
        checks[f"{condition}_mean_safe_success"] = safe / episodes >= args.minimum_mean_safe_success
        checks[f"{condition}_seed_floor"] = min(
            record["safe_success_rate"] for record in subset
        ) >= args.minimum_seed_safe_success
        checks[f"{condition}_violation_rate"] = violations / episodes <= args.maximum_violation_rate
    result = {
        "schema_version": 1,
        "audit": "external_peg_direction_specialists",
        "criteria": {
            "minimum_mean_safe_success": args.minimum_mean_safe_success,
            "minimum_seed_safe_success": args.minimum_seed_safe_success,
            "maximum_violation_rate": args.maximum_violation_rate,
        },
        "conditions": by_condition,
        "checks": checks,
        "specialist_gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.fail_on_reject and not result["specialist_gate_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
