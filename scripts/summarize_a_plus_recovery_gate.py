#!/usr/bin/env python3
"""Aggregate closed-loop manifests against the preregistered A+ gate.

The script deliberately uses safe success (success with no constraint
violation) as its primary endpoint.  It never substitutes raw success.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return (math.nan, math.nan)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return center - radius, center + radius


def parse_method(value: str) -> tuple[str, Path]:
    try:
        name, directory = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("method must be NAME=DIRECTORY") from exc
    return name, Path(directory)


def load_manifests(directory: Path) -> list[dict]:
    manifests = []
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text())
        required = {"episodes", "safe_successes", "violations", "condition"}
        if required <= record.keys():
            manifests.append(record)
    if not manifests:
        raise RuntimeError(f"no closed-loop manifests found in {directory}")
    return manifests


def aggregate(records: list[dict]) -> dict:
    by_condition: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "successes": 0, "safe_successes": 0, "violations": 0}
    )
    for record in records:
        row = by_condition[str(record["condition"])]
        for key in row:
            row[key] += int(record[key])
    total = {key: sum(row[key] for row in by_condition.values()) for key in next(iter(by_condition.values()))}
    for row in [*by_condition.values(), total]:
        row["raw_success_rate"] = row["successes"] / row["episodes"]
        row["safe_success_rate"] = row["safe_successes"] / row["episodes"]
        row["violation_rate"] = row["violations"] / row["episodes"]
        row["safe_success_wilson_95"] = wilson(row["safe_successes"], row["episodes"])
    return {"overall": total, "conditions": dict(sorted(by_condition.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, default=Path("configs/a_plus_recovery_gate_v1.json"))
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--candidate", default="causal")
    parser.add_argument("--oracle", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text())
    criteria = gate["pass_criteria"]
    success_min = criteria.get(
        "closed_loop_success_min", criteria.get("closed_loop_safe_success_min"),
    )
    violation_max = criteria.get(
        "closed_loop_violation_max", criteria.get("violation_rate_max"),
    )
    worst_min = criteria.get(
        "worst_condition_success_min", criteria.get("worst_condition_safe_success_min"),
    )
    if success_min is None or violation_max is None or worst_min is None:
        raise RuntimeError("gate is missing a closed-loop success, violation, or worst-condition threshold")
    method_records: dict[str, list[dict]] = defaultdict(list)
    for name, directory in args.method:
        method_records[name].extend(load_manifests(directory))
    methods = {name: aggregate(records) for name, records in method_records.items()}
    if args.candidate not in methods:
        raise RuntimeError(f"candidate {args.candidate!r} was not supplied")
    candidate = methods[args.candidate]["overall"]
    competitors = {
        name: result for name, result in methods.items()
        if name != args.candidate and name not in set(args.oracle)
    }
    checks = {
        "closed_loop_success": candidate["safe_success_rate"] >= success_min,
        "violation_rate": candidate["violation_rate"] <= violation_max,
        "worst_condition": min(
            row["safe_success_rate"] for row in methods[args.candidate]["conditions"].values()
        ) >= worst_min,
    }
    if "heldout_reverse_safe_success_min" in criteria:
        reverse = methods[args.candidate]["conditions"].get("reverse_ejection")
        if reverse is None:
            raise RuntimeError("candidate manifests omit held-out reverse_ejection")
        checks["heldout_reverse_safe_success"] = (
            reverse["safe_success_rate"]
            >= criteria["heldout_reverse_safe_success_min"]
        )
    if "minimum_independent_training_seeds" in criteria:
        router_seeds = {
            record.get("router_seed")
            for record in method_records[args.candidate]
            if record.get("router_seed") is not None
        }
        checks["training_seed_count"] = (
            len(router_seeds) >= criteria["minimum_independent_training_seeds"]
        )
    comparison = None
    if competitors:
        strongest_name, strongest = max(
            competitors.items(), key=lambda item: item[1]["overall"]["safe_success_rate"]
        )
        baseline = strongest["overall"]
        gain = candidate["safe_success_rate"] - baseline["safe_success_rate"]
        candidate_ci = candidate["safe_success_wilson_95"]
        baseline_ci = baseline["safe_success_wilson_95"]
        # Newcombe's score-interval difference, without a continuity correction.
        difference_ci = (candidate_ci[0] - baseline_ci[1], candidate_ci[1] - baseline_ci[0])
        comparison = {
            "strongest_non_oracle": strongest_name,
            "gain": gain,
            "newcombe_95": difference_ci,
        }
        checks["gain_over_strongest_non_oracle"] = gain >= criteria["gain_over_strongest_non_oracle_min_pp"] / 100
        checks["gain_newcombe_lower"] = difference_ci[0] > criteria["gain_newcombe_95_lower_min_pp"] / 100

    result = {
        "schema_version": 1,
        "primary_endpoint": gate.get(
            "primary_endpoint", "episode_success_without_constraint_violation",
        ),
        "methods": methods,
        "comparison": comparison,
        "checks": checks,
        "primary_gate_pass": all(checks.values()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
