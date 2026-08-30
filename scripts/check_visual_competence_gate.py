#!/usr/bin/env python3
"""Fail closed unless a held-out V3 visual competence gate is satisfied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def check_gate(payload, method, minimum_success, seeds, episodes):
    if payload.get("benchmark_semantics") != "event_reward_intervention_target_only_v3":
        raise ValueError("aggregate does not use V3 event-reward semantics")
    methods = payload.get("conditions", {}).get("nominal", {}).get("methods", [])
    matches = [item for item in methods if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError("nominal aggregate does not contain exactly one selected method")
    result = matches[0]
    if int(result.get("seeds", -1)) != seeds:
        raise ValueError("competence aggregate has the wrong training-seed count")
    if int(result.get("episodes", -1)) != episodes:
        raise ValueError("competence aggregate has the wrong held-out episode count")
    success = float(result["success_rate"])
    return {
        "passed": success >= minimum_success,
        "method": method,
        "training_seeds": seeds,
        "heldout_episodes": episodes,
        "success_rate": success,
        "minimum_success_rate": minimum_success,
        "success_hierarchical_bootstrap_95": result.get(
            "success_hierarchical_bootstrap_95"
        ),
        "constraint_violation_rate": result.get("constraint_violation_rate"),
        "benchmark_semantics": payload["benchmark_semantics"],
    }


def check_visualization_gate(
    payload, method, minimum_raw, minimum_safe, maximum_violation,
    minimum_nominal, seeds=3, episodes=768,
):
    """Require a paper-eligible recovery result before qualitative selection."""
    if payload.get("benchmark_semantics") != "event_reward_intervention_target_only_v3":
        raise ValueError("aggregate does not use V3 event-reward semantics")

    def selected(condition):
        methods = payload.get("conditions", {}).get(condition, {}).get("methods", [])
        matches = [item for item in methods if item.get("method") == method]
        if len(matches) != 1:
            raise ValueError(
                f"{condition} aggregate does not contain exactly one selected method"
            )
        result = matches[0]
        if int(result.get("seeds", -1)) != seeds or int(result.get("episodes", -1)) != episodes:
            raise ValueError(f"{condition} aggregate has the wrong evaluation protocol")
        return result

    forced = selected("intervention")
    nominal = selected("nominal")
    checks = {
        "raw_recovery": float(forced["success_rate"]) >= minimum_raw,
        "safe_recovery": float(forced["safe_success_rate"]) >= minimum_safe,
        "recovery_safety": float(forced["constraint_violation_rate"]) <= maximum_violation,
        "nominal_retention": float(nominal["success_rate"]) >= minimum_nominal,
    }
    return {
        "passed": all(checks.values()), "checks": checks, "method": method,
        "training_seeds": seeds, "heldout_episodes_per_condition": episodes,
        "forced_raw_success_rate": forced["success_rate"],
        "forced_safe_success_rate": forced["safe_success_rate"],
        "forced_constraint_violation_rate": forced["constraint_violation_rate"],
        "nominal_success_rate": nominal["success_rate"],
        "thresholds": {
            "minimum_raw_success_rate": minimum_raw,
            "minimum_safe_success_rate": minimum_safe,
            "maximum_constraint_violation_rate": maximum_violation,
            "minimum_nominal_success_rate": minimum_nominal,
        },
        "benchmark_semantics": payload["benchmark_semantics"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--minimum-success", type=float, default=0.70)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=768)
    args = parser.parse_args()
    result = check_gate(
        json.loads(Path(args.aggregate).read_text(encoding="utf-8")),
        args.method, args.minimum_success, args.seeds, args.episodes,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
