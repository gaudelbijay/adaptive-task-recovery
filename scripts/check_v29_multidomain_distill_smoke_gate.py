#!/usr/bin/env python3
"""Fail-closed development allocation gate for V29 multidomain distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


PROTOCOL = "paired selected-policy causal-head and visual-OOD aggregate"


def load(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def unique(records, variant, condition):
    found = [
        item for item in records
        if item.get("variant") == variant and item.get("condition") == condition
    ]
    if len(found) != 1:
        raise ValueError(f"expected one {variant}/{condition} record")
    return found[0]


def per_seed_rate(record: dict, seed: int) -> float:
    matches = [
        item for item in record.get("per_seed", [])
        if int(item.get("training_seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matched seed-{seed} OOD record")
    return float(matches[0]["variant_safe_success_rate"])


def check(config_path: Path) -> dict:
    config, config_hash = load(config_path)
    incumbent_path = Path(config["incumbent_aggregate"])
    candidate_path = Path(config["candidate_aggregate"])
    incumbent, incumbent_hash = load(incumbent_path)
    candidate, candidate_hash = load(candidate_path)
    if incumbent.get("protocol") != PROTOCOL or candidate.get("protocol") != PROTOCOL:
        raise ValueError("robust gate aggregate protocol mismatch")
    if incumbent.get("training_seeds") != [9351, 4796, 1788]:
        raise ValueError("incumbent seed contract mismatch")
    if candidate.get("training_seeds") != [1788]:
        raise ValueError("candidate smoke must retain seed 1788")
    thresholds = config["thresholds"]
    incumbent_records = incumbent["records"]
    candidate_records = candidate["records"]
    nominal = unique(candidate_records, "baseline", "nominal")
    intervention = unique(candidate_records, "baseline", "intervention")
    causal = unique(candidate_records, "progress_cyclic_shift", "intervention")
    ood_names = sorted({
        item["variant"] for item in incumbent_records
        if item["variant"] != "baseline" and not item["variant"].startswith("progress_")
    })
    if len(ood_names) != 7:
        raise ValueError("expected seven development OOD variants")
    improvements = []
    candidate_rates = []
    matched_seed = 1788
    for name in ood_names:
        for condition in ("nominal", "intervention"):
            base = unique(incumbent_records, name, condition)
            treatment = unique(candidate_records, name, condition)
            base_rate = per_seed_rate(base, matched_seed)
            treatment_rate = per_seed_rate(treatment, matched_seed)
            improvement = treatment_rate - base_rate
            improvements.append(improvement)
            candidate_rates.append(treatment_rate)
    if any(not math.isfinite(value) for value in improvements + candidate_rates):
        raise ValueError("non-finite V29 development statistic")
    mean_improvement = sum(improvements) / len(improvements)
    worst_rate = min(candidate_rates)
    worst_regression = min(improvements)
    causal_interval = causal["paired_cluster_bootstrap_95"]
    checks = {
        "nominal_retention": float(nominal["variant_safe_success_rate"])
        >= float(thresholds["minimum_nominal_baseline_safe_success"]),
        "intervention_retention": float(intervention["variant_safe_success_rate"])
        >= float(thresholds["minimum_intervention_baseline_safe_success"]),
        "mean_ood_improvement": mean_improvement
        >= float(thresholds["minimum_mean_ood_safe_success_improvement"]),
        "worst_ood_safe_success": worst_rate
        >= float(thresholds["minimum_worst_ood_safe_success"]),
        "no_large_ood_regression": worst_regression
        >= -float(thresholds["maximum_individual_ood_regression"]),
        "causal_drop": float(causal["baseline_minus_variant_safe_success"])
        >= float(thresholds["minimum_causal_safe_success_drop"]),
        "causal_lower_bound": (
            not bool(thresholds["require_positive_causal_lower_bound"])
            or float(causal_interval[0]) > 0.0
        ),
    }
    return {
        "schema_version": 1,
        "protocol": "V29 one-seed multidomain development allocation gate",
        "matched_training_seed": matched_seed,
        "candidate_nominal_safe_success": float(nominal["variant_safe_success_rate"]),
        "candidate_intervention_safe_success": float(intervention["variant_safe_success_rate"]),
        "mean_ood_safe_success_improvement": mean_improvement,
        "worst_ood_safe_success": worst_rate,
        "worst_individual_ood_improvement": worst_regression,
        "causal_safe_success_drop": float(causal["baseline_minus_variant_safe_success"]),
        "causal_paired_cluster_bootstrap_95": causal_interval,
        "checks": checks,
        "eligible": all(checks.values()),
        "thresholds": thresholds,
        "source_sha256": {
            str(config_path): config_hash,
            str(incumbent_path): incumbent_hash,
            str(candidate_path): candidate_hash,
            "checker": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "claim_boundary": config["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = check(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
