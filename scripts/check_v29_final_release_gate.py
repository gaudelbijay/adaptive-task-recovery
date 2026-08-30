#!/usr/bin/env python3
"""Fail closed on V29 final retention, safety, causal, and unseen-OOD evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def one(items: list[dict], description: str) -> dict:
    if len(items) != 1:
        raise ValueError(f"expected exactly one {description}, observed {len(items)}")
    return items[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gate_path = Path(args.config)
    gate = load(gate_path)
    standard = load(gate["standard_aggregate"])
    strict = load(gate["strict_aggregate"])
    unseen = load(gate["unseen_aggregate"])
    thresholds = gate["thresholds"]
    method = gate["candidate_method"]

    standard_records = {}
    for condition in ("nominal", "intervention"):
        methods = standard["conditions"][condition]["methods"]
        standard_records[condition] = one(
            [item for item in methods if item["method"] == method],
            f"standard {condition} candidate",
        )
    candidate_strict = one([
        item for item in strict["cohorts"]
        if item["label"] == gate["candidate_strict_label"]
    ], "candidate strict cohort")
    incumbent_strict = one([
        item for item in strict["cohorts"]
        if item["label"] == gate["incumbent_strict_label"]
    ], "incumbent strict cohort")
    ood = [
        item for item in unseen["records"]
        if item["variant"] != "baseline" and not item["variant"].startswith("progress_")
    ]
    if not ood:
        raise ValueError("unseen aggregate lacks OOD records")
    mean_ood = sum(item["variant_safe_success_rate"] for item in ood) / len(ood)
    minimum_ood_seed = min(
        seed["variant_safe_success_rate"] for item in ood for seed in item["per_seed"]
    )
    candidate_strict_drop = (
        incumbent_strict["safe_success_rate"] - candidate_strict["safe_success_rate"]
    )
    checks = {
        "standard_nominal_safe_success": standard_records["nominal"]["safe_success_rate"]
        >= thresholds["minimum_standard_nominal_safe_success"],
        "standard_intervention_safe_success": standard_records["intervention"]["safe_success_rate"]
        >= thresholds["minimum_standard_intervention_safe_success"],
        "standard_per_seed_safe_success": min(
            item["safe_success_rate"]
            for condition in standard_records.values()
            for item in condition["seed_results"]
        ) >= thresholds["minimum_standard_per_seed_safe_success"],
        "strict_safe_success": candidate_strict["safe_success_rate"]
        >= thresholds["minimum_strict_safe_success"],
        "strict_per_seed_safe_success": min(candidate_strict["seed_safe_success_rates"])
        >= thresholds["minimum_strict_per_seed_safe_success"],
        "strict_retention_vs_incumbent": candidate_strict_drop
        <= thresholds["maximum_strict_safe_success_drop_from_incumbent"],
        "mean_unseen_ood_safe_success": mean_ood
        >= thresholds["minimum_mean_unseen_ood_safe_success"],
        "unseen_ood_per_seed_safe_success": minimum_ood_seed
        >= thresholds["minimum_unseen_ood_per_seed_safe_success"],
        "all_frozen_unseen_ood_hypotheses": (
            not thresholds["require_all_frozen_unseen_ood_hypotheses"]
            or unseen["hypotheses"]["selected_policy_is_robust_to_frozen_visual_ood_suite"]
        ),
        "causal_progress_utility": (
            not thresholds["require_causal_progress_utility"]
            or unseen["hypotheses"]["learned_progress_head_has_causal_control_utility"]
        ),
    }
    scalars = [
        standard_records["nominal"]["safe_success_rate"],
        standard_records["intervention"]["safe_success_rate"],
        candidate_strict["safe_success_rate"], candidate_strict_drop,
        mean_ood, minimum_ood_seed,
    ]
    if not all(math.isfinite(float(value)) for value in scalars):
        raise ValueError("V29 final gate contains non-finite evidence")
    payload = {
        "schema_version": 1,
        "gate": gate["name"],
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "standard_nominal_safe_success": standard_records["nominal"]["safe_success_rate"],
            "standard_intervention_safe_success": standard_records["intervention"]["safe_success_rate"],
            "minimum_standard_per_seed_safe_success": min(
                item["safe_success_rate"]
                for condition in standard_records.values()
                for item in condition["seed_results"]
            ),
            "strict_safe_success": candidate_strict["safe_success_rate"],
            "minimum_strict_per_seed_safe_success": min(candidate_strict["seed_safe_success_rates"]),
            "strict_safe_success_drop_from_incumbent": candidate_strict_drop,
            "mean_unseen_ood_safe_success": mean_ood,
            "minimum_unseen_ood_per_seed_safe_success": minimum_ood_seed,
        },
        "thresholds": thresholds,
        "source_sha256": {
            str(gate_path): sha256(gate_path),
            gate["standard_aggregate"]: sha256(gate["standard_aggregate"]),
            gate["strict_aggregate"]: sha256(gate["strict_aggregate"]),
            gate["unseen_aggregate"]: sha256(gate["unseen_aggregate"]),
            "checker": sha256(Path(__file__)),
        },
        "claim_boundary": gate["claim_boundary"],
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
