#!/usr/bin/env python3
"""Fail-closed paired verdict for the V19 continuation temporal-SSL ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_recovery import paired_effect


STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
NOMINAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def one(items: list[dict], key: str, value, description: str) -> dict:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description} for {value!r}")
    return matches[0]


def oriented(comparisons: list[dict], left: str, right: str) -> dict:
    direct = [item for item in comparisons if item["left"] == left and item["right"] == right]
    reverse = [item for item in comparisons if item["left"] == right and item["right"] == left]
    if len(direct) + len(reverse) != 1:
        raise ValueError(f"expected one paired comparison for {left}/{right}")
    result = dict((direct or reverse)[0])
    if reverse:
        result["safe_success_rate_difference"] *= -1
        lo, hi = result["safe_paired_bootstrap_95"]
        result["safe_paired_bootstrap_95"] = [-hi, -lo]
    return {
        "safe_success_rate_difference": result["safe_success_rate_difference"],
        "safe_paired_bootstrap_95": result["safe_paired_bootstrap_95"],
        "paired_training_seeds": result["paired_training_seeds"],
        "paired_episodes": result["paired_episodes"],
    }


def nominal_method(payload: dict, method: str, seeds: list[int], episodes: int) -> dict:
    if payload.get("protocol") != NOMINAL_PROTOCOL:
        raise ValueError("nominal aggregate protocol mismatch")
    record = one(payload.get("conditions", {}).get("nominal", {}).get("methods", []),
                 "method", method, "nominal method")
    if int(record.get("episodes", -1)) != episodes:
        raise ValueError("nominal aggregate episode count mismatch")
    results = sorted(record.get("seed_results", []), key=lambda item: item["training_seed"])
    if [item["training_seed"] for item in results] != sorted(seeds):
        raise ValueError("nominal aggregate training-seed mismatch")
    if any(len(item.get("episode_records", [])) * len(seeds) != episodes for item in results):
        raise ValueError("nominal per-seed episode count mismatch")
    return record


def initial_eval(path: Path) -> tuple[dict, str]:
    """Load the unique step-zero evaluation used to verify paired initialization."""
    raw = path.read_bytes()
    records = []
    for line in raw.splitlines():
        record = json.loads(line)
        if int(record.get("global_step", -1)) == 0 and record.get("eval") is not None:
            records.append(record["eval"])
    if len(records) != 1:
        raise ValueError(f"expected exactly one step-zero evaluation in {path}")
    signature_keys = (
        "success_once", "success_at_end", "constraint_violated", "fail_once",
        "fail_at_end", "goals_completed", "goals_unavailable",
        "visual_progress_bit_accuracy",
    )
    evaluation = records[0]
    missing = [key for key in signature_keys if key not in evaluation]
    if missing:
        raise ValueError(f"step-zero evaluation missing paired fields {missing} in {path}")
    return {key: evaluation[key] for key in signature_keys}, hashlib.sha256(raw).hexdigest()


def compare(config_path: Path) -> dict:
    config, config_hash = load(config_path)
    seeds = [int(seed) for seed in config["required_training_seeds"]]
    episodes = int(config["required_episodes_per_condition"])
    treatment_method, control_method = config["treatment"], config["control"]

    initialization_checks = []
    initialization_hashes = {}
    treatment_metrics = config["treatment_initial_metrics"]
    control_metrics = config["control_initial_metrics"]
    for seed in seeds:
        treatment_path = Path(treatment_metrics.format(seed=seed))
        control_path = Path(control_metrics.format(seed=seed))
        treatment_initial, treatment_initial_hash = initial_eval(treatment_path)
        control_initial, control_initial_hash = initial_eval(control_path)
        if treatment_initial != control_initial:
            raise ValueError(f"step-zero paired initialization mismatch for seed {seed}")
        initialization_checks.append({"training_seed": seed, **treatment_initial})
        initialization_hashes[str(treatment_path)] = treatment_initial_hash
        initialization_hashes[str(control_path)] = control_initial_hash

    strict_path = Path(config["strict_aggregate"])
    strict, strict_hash = load(strict_path)
    if strict.get("protocol") != STRICT_PROTOCOL:
        raise ValueError("strict aggregate protocol mismatch")
    treatment_strict = one(strict.get("cohorts", []), "method", treatment_method,
                           "strict treatment")
    control_strict = one(strict.get("cohorts", []), "method", control_method,
                         "strict control")
    for record in (treatment_strict, control_strict):
        if record.get("training_seeds") != seeds or int(record.get("episodes", -1)) != episodes:
            raise ValueError("strict aggregate seed/episode contract mismatch")
    treatment_label, control_label = treatment_strict["label"], control_strict["label"]

    treatment_nominal_path = Path(config["treatment_nominal_aggregate"])
    control_nominal_path = Path(config["control_nominal_aggregate"])
    treatment_nominal_payload, treatment_nominal_hash = load(treatment_nominal_path)
    control_nominal_payload, control_nominal_hash = load(control_nominal_path)
    treatment_nominal = nominal_method(
        treatment_nominal_payload, treatment_method, seeds, episodes,
    )
    control_nominal = nominal_method(control_nominal_payload, control_method, seeds, episodes)
    left_results = sorted(treatment_nominal["seed_results"], key=lambda item: item["training_seed"])
    right_results = sorted(control_nominal["seed_results"], key=lambda item: item["training_seed"])
    for left, right in zip(left_results, right_results):
        if left.get("batch_seeds") != right.get("batch_seeds"):
            raise ValueError("nominal paired reset seeds mismatch")
    nominal_effect = paired_effect(
        [item["episode_records"] for item in left_results],
        [item["episode_records"] for item in right_results],
        np.random.default_rng(20260829),
    )
    nominal_effect = {
        key: nominal_effect[key] for key in (
            "safe_success_rate_difference", "safe_paired_bootstrap_95",
            "paired_training_seeds", "paired_episodes",
        )
    }

    endpoints = {
        "strict": (
            treatment_strict["safe_success_rate"], control_strict["safe_success_rate"],
            oriented(strict["paired_comparisons"], treatment_label, control_label),
        ),
        "nominal": (
            treatment_nominal["safe_success_rate"], control_nominal["safe_success_rate"],
            nominal_effect,
        ),
        "first_removed": (
            treatment_strict["first_goal_physically_removed"]["safe_success_rate"],
            control_strict["first_goal_physically_removed"]["safe_success_rate"],
            oriented(strict["paired_comparisons_by_branch"]["first_goal_physically_removed"],
                     treatment_label, control_label),
        ),
        "second_removed": (
            treatment_strict["second_goal_physically_removed"]["safe_success_rate"],
            control_strict["second_goal_physically_removed"]["safe_success_rate"],
            oriented(strict["paired_comparisons_by_branch"]["second_goal_physically_removed"],
                     treatment_label, control_label),
        ),
    }
    endpoint_payload = {
        name: {"treatment_safe_success_rate": float(values[0]),
               "control_safe_success_rate": float(values[1]), **values[2]}
        for name, values in endpoints.items()
    }
    primary = min(endpoints, key=lambda name: (float(endpoints[name][1]), name))
    treatment_worst = min(float(values[0]) for values in endpoints.values())
    control_worst = min(float(values[1]) for values in endpoints.values())
    improvement = treatment_worst - control_worst
    checks = {
        "minimum_worst_endpoint_improvement": (
            improvement >= float(config["minimum_worst_endpoint_improvement"])
        ),
        "primary_paired_lower_bound_positive": (
            endpoint_payload[primary]["safe_paired_bootstrap_95"][0] > 0.0
        ),
        "treatment_strict_violation": (
            float(treatment_strict["constraint_violation_rate"])
            <= float(config["maximum_treatment_violation_rate"])
        ),
        "treatment_nominal_violation": (
            float(treatment_nominal["constraint_violation_rate"])
            <= float(config["maximum_treatment_violation_rate"])
        ),
    }
    return {
        "schema_version": 1,
        "protocol": "paired continuation-stage temporal-SSL ablation verdict",
        "config": str(config_path), "config_sha256": config_hash,
        "treatment": treatment_method, "control": control_method,
        "required_training_seeds": seeds,
        "required_episodes_per_condition": episodes,
        "paired_initialization_verified": True,
        "paired_initial_evaluations": initialization_checks,
        "endpoints": endpoint_payload,
        "treatment_worst_endpoint_safe_success_rate": treatment_worst,
        "control_worst_endpoint_safe_success_rate": control_worst,
        "worst_endpoint_improvement": improvement,
        "control_worst_endpoint": primary,
        "checks": checks, "confirmed": all(checks.values()),
        "confirmation_rule": config["confirmation_rule"],
        "claim_boundary": config["claim_boundary"],
        "source_sha256": {
            str(strict_path): strict_hash,
            str(treatment_nominal_path): treatment_nominal_hash,
            str(control_nominal_path): control_nominal_hash,
            "comparator": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **initialization_hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = compare(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
