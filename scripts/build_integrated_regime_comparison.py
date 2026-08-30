#!/usr/bin/env python3
"""Build the paper table/figure for strict recovery and nominal retention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEMANTICS = "event_reward_intervention_target_only_v3"
STRICT_PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"
VISUAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"
STATE_PROTOCOL = "held-out deterministic state-policy evaluation"


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def one(items: list[dict], key: str, value: str, description: str) -> dict:
    matches = [item for item in items if item.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"expected one {description} {value!r}")
    return matches[0]


def validate_count(
    record: dict, *, required_episodes: int, required_seeds: int,
    state: bool = False,
) -> None:
    if int(record.get("episodes", -1)) != required_episodes:
        raise ValueError(
            f"integrated comparison requires exactly {required_episodes} episodes"
        )
    seeds = record.get("seeds") if state else record.get("training_seeds")
    count = int(seeds) if state else len(seeds or [])
    if count != required_seeds:
        raise ValueError(
            f"integrated comparison requires exactly {required_seeds} training seeds"
        )


def nominal_record(
    spec: dict, *, required_episodes: int, required_seeds: int,
) -> tuple[dict, str]:
    path = Path(spec["nominal_aggregate"])
    payload, digest = load(path)
    if payload.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("nominal aggregate has wrong benchmark semantics")
    if spec["kind"] == "visual":
        if payload.get("protocol") != VISUAL_PROTOCOL:
            raise ValueError("visual nominal aggregate has wrong protocol")
        record = one(
            payload.get("conditions", {}).get("nominal", {}).get("methods", []),
            "method", spec["method"], "visual nominal method",
        )
        validate_count(
            record, required_episodes=required_episodes,
            required_seeds=required_seeds, state=True,
        )
    elif spec["kind"] == "state":
        if payload.get("protocol") != STATE_PROTOCOL:
            raise ValueError("state nominal aggregate has wrong protocol")
        record = one(
            payload.get("nominal_condition", []), "method", spec["method"],
            "state nominal method",
        )
        validate_count(
            record, required_episodes=required_episodes,
            required_seeds=required_seeds, state=True,
        )
    else:
        raise ValueError(f"unknown cohort kind: {spec['kind']}")
    return record, digest


def nominal_safe_success(record: dict, kind: str) -> float:
    key = "safe_success_rate" if kind == "visual" else "pooled_safe_success_rate"
    return rate(record, key, f"{kind} nominal safe success")


def rate(record: dict, key: str, description: str) -> float:
    if key not in record:
        raise ValueError(f"{description} is missing {key}")
    value = float(record[key])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{description} must be a finite rate in [0, 1]")
    return value


def interval(record: dict, key: str, description: str) -> list[float]:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{description} must contain two endpoints")
    bounds = [float(value) for value in values]
    if (
        not all(math.isfinite(value) for value in bounds)
        or not 0.0 <= bounds[0] <= bounds[1] <= 1.0
    ):
        raise ValueError(f"{description} must be an ordered interval in [0, 1]")
    return bounds


def err(center: float, interval: list[float]) -> list[float]:
    return [max(0.0, center - interval[0]), max(0.0, interval[1] - center)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config, config_hash = load(config_path)
    strict_path = Path(config["strict_aggregate"])
    required_episodes = int(config.get("required_episodes", 768))
    required_seeds = int(config.get("required_training_seeds", 3))
    if required_episodes <= 0 or required_seeds <= 0:
        raise ValueError("required episode and seed counts must be positive")
    strict, strict_hash = load(strict_path)
    if strict.get("protocol") != STRICT_PROTOCOL:
        raise ValueError("strict aggregate has wrong protocol")
    if strict.get("benchmark_semantics") != SEMANTICS:
        raise ValueError("strict aggregate has wrong benchmark semantics")

    rows = []
    hashes = {str(strict_path): strict_hash}
    for spec in config["cohorts"]:
        strict_record = one(
            strict.get("cohorts", []), "label", spec["label"], "strict cohort"
        )
        if strict_record.get("kind") != spec["kind"]:
            raise ValueError("strict cohort kind mismatch")
        if strict_record.get("method") != spec["method"]:
            raise ValueError("strict cohort method mismatch")
        validate_count(
            strict_record, required_episodes=required_episodes,
            required_seeds=required_seeds,
        )
        nominal, nominal_hash = nominal_record(
            spec, required_episodes=required_episodes,
            required_seeds=required_seeds,
        )
        hashes[spec["nominal_aggregate"]] = nominal_hash
        first = strict_record["first_goal_physically_removed"]
        second = strict_record["second_goal_physically_removed"]
        row = {
            "label": spec["label"], "kind": spec["kind"],
            "method": spec["method"],
            "episodes_per_condition": required_episodes,
            "strict_safe_success_rate": rate(
                strict_record, "safe_success_rate", "strict safe success"
            ),
            "strict_safe_hierarchical_95": interval(
                strict_record, "safe_success_hierarchical_bootstrap_95",
                "strict safe-success interval",
            ),
            "nominal_safe_success_rate": nominal_safe_success(
                nominal, spec["kind"]
            ),
            "nominal_safe_hierarchical_95": interval(
                nominal, "safe_success_hierarchical_bootstrap_95",
                "nominal safe-success interval",
            ),
            "first_removed_safe_success_rate": rate(
                first, "safe_success_rate", "first-removed safe success"
            ),
            "first_removed_safe_hierarchical_95": interval(
                first, "safe_success_hierarchical_bootstrap_95",
                "first-removed safe-success interval",
            ),
            "second_removed_safe_success_rate": rate(
                second, "safe_success_rate", "second-removed safe success"
            ),
            "second_removed_safe_hierarchical_95": interval(
                second, "safe_success_hierarchical_bootstrap_95",
                "second-removed safe-success interval",
            ),
            "strict_violation_rate": rate(
                strict_record, "constraint_violation_rate",
                "strict constraint violations",
            ),
            "nominal_violation_rate": rate(
                nominal, "constraint_violation_rate",
                "nominal constraint violations",
            ),
        }
        row["worst_case_safe_success_rate"] = min(
            row["strict_safe_success_rate"], row["nominal_safe_success_rate"],
            row["first_removed_safe_success_rate"],
            row["second_removed_safe_success_rate"],
        )
        rows.append(row)

    payload = {
        "schema_version": 1,
        "protocol": "matched strict-recovery and nominal-retention comparison",
        "benchmark_semantics": SEMANTICS, "experiment": config["name"],
        "required_training_seeds": required_seeds,
        "required_episodes_per_condition": required_episodes,
        "config": str(config_path), "config_sha256": config_hash,
        "source_sha256": hashes, "cohorts": rows,
        "claim_boundary": config["claim_boundary"],
    }
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    temporary = json_path.with_name(f".{json_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, json_path)

    header = [
        "Cohort", "Kind", "Strict safe [hier. 95%]",
        "Nominal safe [hier. 95%]", "First removed safe [hier. 95%]",
        "Second removed safe [hier. 95%]", "Worst safe", "Strict violation",
        "Nominal violation",
    ]
    def rate_with_interval(value: float, bounds: list[float]) -> str:
        return (
            f"{100 * value:.2f}% "
            f"[{100 * bounds[0]:.2f}, {100 * bounds[1]:.2f}]"
        )

    table_rows = [[
        item["label"], item["kind"],
        rate_with_interval(
            item["strict_safe_success_rate"],
            item["strict_safe_hierarchical_95"],
        ),
        rate_with_interval(
            item["nominal_safe_success_rate"],
            item["nominal_safe_hierarchical_95"],
        ),
        rate_with_interval(
            item["first_removed_safe_success_rate"],
            item["first_removed_safe_hierarchical_95"],
        ),
        rate_with_interval(
            item["second_removed_safe_success_rate"],
            item["second_removed_safe_hierarchical_95"],
        ),
        f"{100 * item['worst_case_safe_success_rate']:.2f}%",
        f"{100 * item['strict_violation_rate']:.2f}%",
        f"{100 * item['nominal_violation_rate']:.2f}%",
    ] for item in rows]
    markdown = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
        *("| " + " | ".join(row) + " |" for row in table_rows),
    ]
    output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as handle:
        csv.writer(handle).writerows([header, *table_rows])

    labels = [item["label"].replace("_", " ").title() for item in rows]
    x = np.arange(len(rows)); width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax in axes:
        ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    strict_safe = np.array([item["strict_safe_success_rate"] for item in rows])
    nominal_safe = np.array([item["nominal_safe_success_rate"] for item in rows])
    axes[0].bar(x-width/2, strict_safe, width, label="Strict", color="#3977b8")
    axes[0].bar(x+width/2, nominal_safe, width, label="Nominal", color="#7fb36d")
    axes[0].errorbar(x-width/2, strict_safe, yerr=np.array([err(v,r["strict_safe_hierarchical_95"]) for v,r in zip(strict_safe,rows)]).T, fmt="none", color="black", capsize=3)
    axes[0].errorbar(x+width/2, nominal_safe, yerr=np.array([err(v,r["nominal_safe_hierarchical_95"]) for v,r in zip(nominal_safe,rows)]).T, fmt="none", color="black", capsize=3)
    axes[0].set_title("A. Same-policy retention"); axes[0].legend()
    first = np.array([r["first_removed_safe_success_rate"] for r in rows]); second=np.array([r["second_removed_safe_success_rate"] for r in rows])
    axes[1].bar(x-width/2, first, width, label="First removed", color="#8055a8")
    axes[1].bar(x+width/2, second, width, label="Second removed", color="#4b9a68")
    axes[1].errorbar(x-width/2, first, yerr=np.array([err(v,r["first_removed_safe_hierarchical_95"]) for v,r in zip(first,rows)]).T, fmt="none", color="black", capsize=3)
    axes[1].errorbar(x+width/2, second, yerr=np.array([err(v,r["second_removed_safe_hierarchical_95"]) for v,r in zip(second,rows)]).T, fmt="none", color="black", capsize=3)
    axes[1].set_title("B. Strict recovery branches"); axes[1].legend()
    sv=np.array([r["strict_violation_rate"] for r in rows]); nv=np.array([r["nominal_violation_rate"] for r in rows])
    axes[2].bar(x-width/2, sv, width, label="Strict", color="#d17c2f")
    axes[2].bar(x+width/2, nv, width, label="Nominal", color="#e3ad71")
    axes[2].set_title("C. Constraint violations"); axes[2].legend()
    for ax in axes:
        ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Rate")
    fig.suptitle("Matched integrated recovery and retention", fontsize=14)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
