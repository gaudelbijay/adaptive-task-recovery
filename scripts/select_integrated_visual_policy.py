#!/usr/bin/env python3
"""Fail-closed selection for a single policy that retains both task regimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
NOMINAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"
SEMANTICS = "event_reward_intervention_target_only_v3"


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def unique(items: list[dict], key: str, value: str, description: str) -> dict:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description}: {value}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config, config_hash = load(config_path)
    strict_path = Path(config["strict_aggregate"])
    strict, strict_hash = load(strict_path)
    if strict.get("protocol") != STRICT_PROTOCOL:
        raise ValueError("strict aggregate has the wrong protocol")
    if strict.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("strict aggregate has the wrong benchmark semantics")
    thresholds = config["thresholds"]
    records = []
    source_hashes = {str(strict_path): strict_hash}

    for candidate in config["candidates"]:
        strict_record = unique(
            strict.get("cohorts", []), "label", candidate["label"], "strict cohort"
        )
        if strict_record.get("method") != candidate["method"]:
            raise ValueError("strict cohort method does not match selection config")
        if int(strict_record.get("episodes", -1)) != 768:
            raise ValueError("strict cohort must contain exactly 768 episodes")
        if len(strict_record.get("training_seeds", [])) != 3:
            raise ValueError("strict cohort must contain exactly three training seeds")
        nominal_path = Path(candidate["nominal_aggregate"])
        nominal, nominal_hash = load(nominal_path)
        source_hashes[str(nominal_path)] = nominal_hash
        if nominal.get("protocol") != NOMINAL_PROTOCOL:
            raise ValueError("nominal aggregate has the wrong protocol")
        if nominal.get("benchmark_semantics") != SEMANTICS:
            raise ValueError("nominal aggregate has the wrong benchmark semantics")
        nominal_condition = nominal.get("conditions", {}).get("nominal", {})
        nominal_record = unique(
            nominal_condition.get("methods", []), "method", candidate["method"],
            "nominal method",
        )
        if int(nominal_record.get("episodes", -1)) != 768:
            raise ValueError("nominal method must contain exactly 768 episodes")
        if int(nominal_record.get("seeds", -1)) != 3:
            raise ValueError("nominal method must contain exactly three training seeds")

        strict_safe = float(strict_record["safe_success_rate"])
        nominal_safe = float(nominal_record["safe_success_rate"])
        first_safe = float(
            strict_record["first_goal_physically_removed"]["safe_success_rate"]
        )
        second_safe = float(
            strict_record["second_goal_physically_removed"]["safe_success_rate"]
        )
        strict_violation = float(strict_record["constraint_violation_rate"])
        nominal_violation = float(nominal_record["constraint_violation_rate"])
        checks = {
            "strict_safe": strict_safe >= thresholds["minimum_strict_safe"],
            "nominal_safe": nominal_safe >= thresholds["minimum_nominal_safe"],
            "first_removed_safe": (
                first_safe >= thresholds["minimum_first_removed_safe"]
            ),
            "second_removed_safe": (
                second_safe >= thresholds["minimum_second_removed_safe"]
            ),
            "strict_violation": (
                strict_violation <= thresholds["maximum_strict_violation"]
            ),
            "nominal_violation": (
                nominal_violation <= thresholds["maximum_nominal_violation"]
            ),
        }
        records.append({
            "label": candidate["label"], "method": candidate["method"],
            "eligible": all(checks.values()), "checks": checks,
            "strict_safe_success_rate": strict_safe,
            "nominal_safe_success_rate": nominal_safe,
            "first_removed_safe_success_rate": first_safe,
            "second_removed_safe_success_rate": second_safe,
            "strict_violation_rate": strict_violation,
            "nominal_violation_rate": nominal_violation,
            "worst_case_safe_success_rate": min(
                strict_safe, nominal_safe, first_safe, second_safe
            ),
        })

    eligible = [item for item in records if item["eligible"]]
    eligible.sort(
        key=lambda item: (
            item["worst_case_safe_success_rate"],
            item["strict_safe_success_rate"],
            item["nominal_safe_success_rate"],
        ),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "protocol": "predeclared integrated visual-policy selection",
        "experiment": config["name"],
        "config": str(config_path), "config_sha256": config_hash,
        "source_sha256": source_hashes, "thresholds": thresholds,
        "candidates": records,
        "selected": eligible[0]["label"] if eligible else None,
        "all_candidates_ineligible": not eligible,
        "ranking_metric": "minimum of strict, nominal, first-removal, and second-removal safe success",
        "claim_boundary": config["claim_boundary"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
