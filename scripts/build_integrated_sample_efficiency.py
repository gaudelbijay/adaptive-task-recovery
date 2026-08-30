#!/usr/bin/env python3
"""Join matched held-out outcomes with validated per-method interaction costs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


PERFORMANCE_PROTOCOL = "matched strict-recovery and nominal-retention comparison"
CONTRACT_PROTOCOL = "configuration-derived method information and interaction accounting"


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def one(items: list[dict], method: str, source: str) -> dict:
    matches = [item for item in items if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {source} record for {method!r}")
    return matches[0]


def nonnegative_int(record: dict, key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"method contract field {key!r} must be a nonnegative integer")
    return value


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def build(performance_path: Path, contract_path: Path) -> dict:
    performance, performance_hash = load(performance_path)
    contract, contract_hash = load(contract_path)
    if performance.get("protocol") != PERFORMANCE_PROTOCOL:
        raise ValueError("performance report protocol mismatch")
    if contract.get("protocol") != CONTRACT_PROTOCOL:
        raise ValueError("method-information protocol mismatch")
    required_seeds = int(performance.get("required_training_seeds", -1))
    if required_seeds <= 0:
        raise ValueError("performance report has invalid training-seed count")

    rows = []
    for result in performance.get("cohorts", []):
        method = result["method"]
        info = one(contract.get("methods", []), method, "method-information")
        seeds = info.get("training_seeds")
        if not isinstance(seeds, list) or len(seeds) != required_seeds:
            raise ValueError(f"method-information seed count mismatch for {method}")
        ppo = nonnegative_int(info, "executed_ppo_interactions_per_seed")
        dagger = nonnegative_int(info, "dagger_interactions_per_seed")
        per_seed = nonnegative_int(info, "new_interactions_per_seed")
        all_seeds = nonnegative_int(info, "new_interactions_all_seeds")
        if ppo + dagger != per_seed:
            raise ValueError(f"interaction arithmetic mismatch for {method}")
        if per_seed * required_seeds != all_seeds:
            raise ValueError(f"all-seed interaction arithmetic mismatch for {method}")
        excludes_upstream = info.get("reported_interactions_exclude_upstream_training")
        if not isinstance(excludes_upstream, bool):
            raise ValueError(f"upstream-training boundary missing for {method}")
        has_upstream = any(info.get(key) for key in (
            "initializer_checkpoint", "teacher_checkpoint",
            "nominal_visual_teacher_checkpoint", "strict_state_teacher_checkpoint",
        ))
        if has_upstream and not excludes_upstream:
            raise ValueError(f"upstream training is not excluded explicitly for {method}")
        rows.append({
            "label": result["label"],
            "method": method,
            "modality": info["modality"],
            "deployed_actor_inputs": info["deployed_actor_inputs"],
            "training_only_asymmetric_critic": bool(
                info["training_only_asymmetric_critic"]
            ),
            "training_only_state_teacher": bool(info["training_only_state_teacher"]),
            "training_only_goal_resolution_labels": bool(
                info["training_only_goal_resolution_labels"]
            ),
            "ppo_interactions_per_seed": ppo,
            "dagger_interactions_per_seed": dagger,
            "new_interactions_per_seed": per_seed,
            "new_interactions_all_seeds": all_seeds,
            "reported_interactions_exclude_upstream_training": excludes_upstream,
            "strict_safe_success_rate": result["strict_safe_success_rate"],
            "nominal_safe_success_rate": result["nominal_safe_success_rate"],
            "worst_case_safe_success_rate": result["worst_case_safe_success_rate"],
            "strict_violation_rate": result["strict_violation_rate"],
            "nominal_violation_rate": result["nominal_violation_rate"],
        })
    if not rows:
        raise ValueError("sample-efficiency report has no cohorts")
    return {
        "schema_version": 1,
        "protocol": "matched outcome and new-stage interaction accounting",
        "required_training_seeds": required_seeds,
        "source_sha256": {
            str(performance_path): performance_hash,
            str(contract_path): contract_hash,
        },
        "rows": rows,
        "claim_boundary": (
            "Interaction counts cover each method's reported new training stage. Where "
            "a method has an initializer or teacher, upstream training is excluded and "
            "disclosed by the method-information contract. Held-out rates are not normalized into an "
            "invented cross-method efficiency score; this is simulation-only accounting."
        ),
    }


def write(payload: dict, prefix: Path) -> None:
    atomic_text(prefix.with_suffix(".json"), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(payload["rows"][0])
    csv_path = prefix.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(payload["rows"])
    os.replace(temporary, csv_path)
    lines = [
        "# Matched outcome and interaction accounting",
        "",
        "| Method | Modality | PPO/seed | DAgger/seed | New/seed | Strict safe | Nominal safe | Worst safe |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['label']} | {row['modality']} | {row['ppo_interactions_per_seed']:,} "
            f"| {row['dagger_interactions_per_seed']:,} | {row['new_interactions_per_seed']:,} "
            f"| {100 * row['strict_safe_success_rate']:.2f}% "
            f"| {100 * row['nominal_safe_success_rate']:.2f}% "
            f"| {100 * row['worst_case_safe_success_rate']:.2f}% |"
        )
    lines.extend(["", payload["claim_boundary"], ""])
    atomic_text(prefix.with_suffix(".md"), "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", required=True, type=Path)
    parser.add_argument("--method-contract", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    write(build(args.performance, args.method_contract), args.output_prefix)


if __name__ == "__main__":
    main()
