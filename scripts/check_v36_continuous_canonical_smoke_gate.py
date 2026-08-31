#!/usr/bin/env python3
"""Fail closed on V36 retention, causality, and development-domain breadth."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def exactly_one(items, description):
    if len(items) != 1:
        raise ValueError(f"expected one {description}, observed {len(items)}")
    return items[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gate_path = Path(args.config)
    gate = load(gate_path)
    aggregate_path = Path(gate["candidate_aggregate"])
    aggregate = load(aggregate_path)
    if aggregate.get("method") != gate["candidate_method"]:
        raise ValueError("V36 gate candidate identity mismatch")
    if aggregate.get("training_seeds") != [int(gate["matched_training_seed"])]:
        raise ValueError("V36 gate requires the frozen one-seed smoke")
    baseline = {
        condition: exactly_one([
            item for item in aggregate["records"]
            if item["variant"] == "baseline" and item["condition"] == condition
        ], f"{condition} baseline")
        for condition in ("nominal", "intervention")
    }
    causal = exactly_one([
        item for item in aggregate["records"]
        if item["variant"] == "progress_cyclic_shift"
        and item["condition"] == "intervention"
    ], "causal record")
    ood = [
        item for item in aggregate["records"]
        if item["variant"] not in ("baseline", "progress_cyclic_shift")
    ]
    if not ood:
        raise ValueError("V36 gate lacks development OOD records")
    mean_ood = sum(item["variant_safe_success_rate"] for item in ood) / len(ood)
    worst_ood = min(item["variant_safe_success_rate"] for item in ood)
    thresholds = gate["thresholds"]
    checks = {
        "nominal_retention": baseline["nominal"]["variant_safe_success_rate"]
        >= thresholds["minimum_nominal_safe_success"],
        "intervention_retention": baseline["intervention"]["variant_safe_success_rate"]
        >= thresholds["minimum_intervention_safe_success"],
        "causal_drop": causal["baseline_minus_variant_safe_success"]
        >= thresholds["minimum_causal_safe_success_drop"],
        "causal_lower_bound": (
            not thresholds["require_positive_causal_lower_bound"]
            or causal["paired_cluster_bootstrap_95"][0] > 0
        ),
        "mean_development_ood": mean_ood
        >= thresholds["minimum_mean_development_ood_safe_success"],
        "worst_development_ood": worst_ood
        >= thresholds["minimum_worst_development_ood_safe_success"],
    }
    payload = {
        "schema_version": 1, "gate": gate["name"], "eligible": all(checks.values()),
        "checks": checks, "observed": {
            "nominal_safe_success": baseline["nominal"]["variant_safe_success_rate"],
            "intervention_safe_success": baseline["intervention"]["variant_safe_success_rate"],
            "causal_safe_success_drop": causal["baseline_minus_variant_safe_success"],
            "causal_paired_cluster_bootstrap_95": causal["paired_cluster_bootstrap_95"],
            "mean_development_ood_safe_success": mean_ood,
            "worst_development_ood_safe_success": worst_ood,
        },
        "thresholds": thresholds, "source_sha256": {
            str(gate_path): sha256(gate_path), str(aggregate_path): sha256(aggregate_path),
            "checker": sha256(Path(__file__)),
        }, "claim_boundary": gate["claim_boundary"],
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
