#!/usr/bin/env python3
"""Fail closed unless both routed DAgger specialists pass their own regime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


NOMINAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"
STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
SEMANTICS = "event_reward_intervention_target_only_v3"


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def unique(items, key, value, description):
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description}: {value}")
    return matches[0]


def check(config):
    nominal_path = Path(config["nominal_aggregate"])
    strict_path = Path(config["strict_aggregate"])
    nominal, nominal_hash = load(nominal_path)
    strict, strict_hash = load(strict_path)
    if nominal.get("protocol") != NOMINAL_PROTOCOL:
        raise ValueError("nominal specialist has the wrong protocol")
    if strict.get("protocol") != STRICT_PROTOCOL:
        raise ValueError("strict specialist has the wrong protocol")
    if nominal.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("nominal specialist has the wrong semantics")
    if strict.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("strict specialist has the wrong semantics")
    nominal_record = unique(
        nominal.get("conditions", {}).get("nominal", {}).get("methods", []),
        "method", config["nominal_method"], "nominal method",
    )
    strict_record = unique(
        strict.get("cohorts", []), "label", config["strict_label"],
        "strict cohort",
    )
    if strict_record.get("method") != config["strict_method"]:
        raise ValueError("strict specialist method mismatch")
    seeds = [int(seed) for seed in config["seeds"]]
    episodes = int(config["episodes_per_condition"])
    if int(nominal_record.get("episodes", -1)) != episodes:
        raise ValueError("nominal specialist has the wrong episode count")
    if int(nominal_record.get("seeds", -1)) != len(seeds):
        raise ValueError("nominal specialist has the wrong seed count")
    if int(strict_record.get("episodes", -1)) != episodes:
        raise ValueError("strict specialist has the wrong episode count")
    if [int(seed) for seed in strict_record.get("training_seeds", [])] != seeds:
        raise ValueError("strict specialist has the wrong training seeds")

    metrics = {
        "nominal_raw_success_rate": float(nominal_record["success_rate"]),
        "nominal_safe_success_rate": float(nominal_record["safe_success_rate"]),
        "strict_safe_success_rate": float(strict_record["safe_success_rate"]),
        "first_removed_safe_success_rate": float(
            strict_record["first_goal_physically_removed"]["safe_success_rate"]
        ),
        "second_removed_safe_success_rate": float(
            strict_record["second_goal_physically_removed"]["safe_success_rate"]
        ),
        "nominal_violation_rate": float(nominal_record["constraint_violation_rate"]),
        "strict_violation_rate": float(strict_record["constraint_violation_rate"]),
    }
    thresholds = config["thresholds"]
    checks = {
        "nominal_raw": metrics["nominal_raw_success_rate"] >= thresholds["minimum_nominal_raw"],
        "nominal_safe": metrics["nominal_safe_success_rate"] >= thresholds["minimum_nominal_safe"],
        "strict_safe": metrics["strict_safe_success_rate"] >= thresholds["minimum_strict_safe"],
        "first_removed_safe": metrics["first_removed_safe_success_rate"] >= thresholds["minimum_first_removed_safe"],
        "second_removed_safe": metrics["second_removed_safe_success_rate"] >= thresholds["minimum_second_removed_safe"],
        "nominal_violation": metrics["nominal_violation_rate"] <= thresholds["maximum_nominal_violation"],
        "strict_violation": metrics["strict_violation_rate"] <= thresholds["maximum_strict_violation"],
    }
    return {
        "schema_version": 1,
        "protocol": "frozen dual-specialist DAgger teacher allocation gate",
        "nominal_method": config["nominal_method"],
        "strict_method": config["strict_method"],
        "training_seeds": seeds,
        "episodes_per_condition": episodes,
        **metrics,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "source_sha256": {
            str(nominal_path): nominal_hash,
            str(strict_path): strict_hash,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    payload = check(json.loads(config_bytes))
    payload["config"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
