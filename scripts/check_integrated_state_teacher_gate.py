#!/usr/bin/env python3
"""Fail closed unless the matched state teacher is strong in both regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
STATE_PROTOCOL = "held-out deterministic state-policy evaluation"
SEMANTICS = "event_reward_intervention_target_only_v3"


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _unique(items: list[dict], key: str, value: str, description: str) -> dict:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description}: {value}")
    return matches[0]


def check(config: dict) -> dict:
    strict_path = Path(config["strict_aggregate"])
    nominal_path = Path(config["state_aggregate"])
    strict, strict_hash = _load(strict_path)
    nominal, nominal_hash = _load(nominal_path)
    if strict.get("protocol") != STRICT_PROTOCOL:
        raise ValueError("strict state-teacher aggregate has the wrong protocol")
    if nominal.get("protocol") != STATE_PROTOCOL:
        raise ValueError("state-teacher aggregate has the wrong protocol")
    if strict.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("strict state-teacher aggregate has the wrong semantics")
    if nominal.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("state-teacher aggregate has the wrong semantics")
    if nominal.get("experiment") != config["expected_experiment"]:
        raise ValueError("state-teacher aggregate has the wrong experiment")

    strict_record = _unique(
        strict.get("cohorts", []), "label", config["strict_label"], "strict cohort"
    )
    nominal_record = _unique(
        nominal.get("nominal_condition", []), "method", config["method"],
        "nominal method",
    )
    configured_record = _unique(
        nominal.get("environments", []), "method", config["method"],
        "configured-distribution method",
    )
    if strict_record.get("method") != config["method"]:
        raise ValueError("strict state-teacher method mismatch")
    seeds = [int(seed) for seed in config["seeds"]]
    expected_episodes = int(config["episodes_per_condition"])
    if [int(seed) for seed in strict_record.get("training_seeds", [])] != seeds:
        raise ValueError("strict state teacher has the wrong training seeds")
    for record, description in (
        (strict_record, "strict"),
        (nominal_record, "nominal"),
        (configured_record, "configured-distribution"),
    ):
        if int(record.get("episodes", -1)) != expected_episodes:
            raise ValueError(f"{description} state teacher has the wrong episode count")
        observed_seeds = record.get("seeds", len(record.get("training_seeds", [])))
        if int(observed_seeds) != len(seeds):
            raise ValueError(f"{description} state teacher has the wrong seed count")

    strict_safe = float(strict_record["safe_success_rate"])
    nominal_safe = float(nominal_record["pooled_safe_success_rate"])
    first_safe = float(
        strict_record["first_goal_physically_removed"]["safe_success_rate"]
    )
    second_safe = float(
        strict_record["second_goal_physically_removed"]["safe_success_rate"]
    )
    strict_violation = float(strict_record["constraint_violation_rate"])
    nominal_violation = float(nominal_record["constraint_violation_rate"])
    thresholds = config["thresholds"]
    checks = {
        "strict_safe": strict_safe >= thresholds["minimum_strict_safe"],
        "nominal_safe": nominal_safe >= thresholds["minimum_nominal_safe"],
        "first_removed_safe": first_safe >= thresholds["minimum_first_removed_safe"],
        "second_removed_safe": second_safe >= thresholds["minimum_second_removed_safe"],
        "strict_violation": strict_violation <= thresholds["maximum_strict_violation"],
        "nominal_violation": nominal_violation <= thresholds["maximum_nominal_violation"],
    }
    return {
        "schema_version": 1,
        "protocol": "predeclared integrated state-teacher allocation gate",
        "method": config["method"],
        "training_seeds": seeds,
        "episodes_per_condition": expected_episodes,
        "strict_safe_success_rate": strict_safe,
        "nominal_safe_success_rate": nominal_safe,
        "first_removed_safe_success_rate": first_safe,
        "second_removed_safe_success_rate": second_safe,
        "strict_violation_rate": strict_violation,
        "nominal_violation_rate": nominal_violation,
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
        "source_sha256": {
            str(strict_path): strict_hash,
            str(nominal_path): nominal_hash,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
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
