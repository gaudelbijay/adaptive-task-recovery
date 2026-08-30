#!/usr/bin/env python3
"""Render paper-ready strict-removal tables and figures from one aggregate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROTOCOL = "held-out deterministic strict-actual-removal policy evaluation"


def _atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _label(value: str) -> str:
    return value.replace("_", " ").replace("adaptive", "adapt.").title()


def _interval_errors(center: float, interval: list[float]) -> list[float]:
    return [max(0.0, center - interval[0]), max(0.0, interval[1] - center)]


def _validate(payload: dict) -> list[dict]:
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("strict figure input has the wrong protocol")
    cohorts = payload.get("cohorts")
    if not isinstance(cohorts, list) or len(cohorts) < 2:
        raise ValueError("strict figure requires at least two complete cohorts")
    required = {
        "label", "kind", "episodes", "success_rate", "safe_success_rate",
        "success_hierarchical_bootstrap_95",
        "safe_success_hierarchical_bootstrap_95", "constraint_violation_rate",
        "constraint_violation_hierarchical_bootstrap_95",
        "first_goal_physically_removed", "second_goal_physically_removed",
    }
    for cohort in cohorts:
        missing = required - cohort.keys()
        if missing:
            raise ValueError(f"strict cohort lacks fields: {sorted(missing)}")
        if int(cohort["episodes"]) <= 0:
            raise ValueError("strict cohort has no episodes")
        for branch in (
            "first_goal_physically_removed", "second_goal_physically_removed",
        ):
            if int(cohort[branch].get("episodes", 0)) <= 0:
                raise ValueError("strict branch has no episodes")
    return cohorts


def _table(cohorts: list[dict]) -> tuple[str, list[list[str]]]:
    header = [
        "Cohort", "Kind", "Episodes", "Raw success", "Raw hierarchical 95%",
        "Safe success", "Safe hierarchical 95%", "Violation",
        "First removed safe", "Second removed safe",
    ]
    rows = []
    for item in cohorts:
        raw_ci = item["success_hierarchical_bootstrap_95"]
        safe_ci = item["safe_success_hierarchical_bootstrap_95"]
        rows.append([
            item["label"], item["kind"], str(item["episodes"]),
            f"{100 * item['success_rate']:.2f}%",
            f"[{100 * raw_ci[0]:.2f}, {100 * raw_ci[1]:.2f}]",
            f"{100 * item['safe_success_rate']:.2f}%",
            f"[{100 * safe_ci[0]:.2f}, {100 * safe_ci[1]:.2f}]",
            f"{100 * item['constraint_violation_rate']:.2f}%",
            f"{100 * item['first_goal_physically_removed']['safe_success_rate']:.2f}%",
            f"{100 * item['second_goal_physically_removed']['safe_success_rate']:.2f}%",
        ])
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n", [header, *rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    aggregate_path = Path(args.aggregate)
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    cohorts = _validate(payload)
    output = Path(args.output_prefix)
    output.parent.mkdir(parents=True, exist_ok=True)

    markdown, csv_rows = _table(cohorts)
    _atomic_text(markdown, output.with_suffix(".md"))
    csv_path = output.with_suffix(".csv")
    temporary_csv = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(csv_rows)
    os.replace(temporary_csv, csv_path)

    labels = [_label(item["label"]) for item in cohorts]
    x = np.arange(len(cohorts))
    width = 0.36
    colors = ["#3977b8" if item["kind"] == "visual" else "#d17c2f" for item in cohorts]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)

    raw = np.asarray([item["success_rate"] for item in cohorts])
    safe = np.asarray([item["safe_success_rate"] for item in cohorts])
    raw_err = np.asarray([
        _interval_errors(value, item["success_hierarchical_bootstrap_95"])
        for value, item in zip(raw, cohorts)
    ]).T
    safe_err = np.asarray([
        _interval_errors(value, item["safe_success_hierarchical_bootstrap_95"])
        for value, item in zip(safe, cohorts)
    ]).T
    axes[0].bar(x - width / 2, raw, width, color=colors, alpha=0.95, label="Raw")
    axes[0].bar(
        x + width / 2, safe, width, color=colors, alpha=0.48,
        edgecolor=colors, linewidth=1.2, label="Safe",
    )
    axes[0].errorbar(x - width / 2, raw, yerr=raw_err, fmt="none", ecolor="black", capsize=3)
    axes[0].errorbar(x + width / 2, safe, yerr=safe_err, fmt="none", ecolor="black", capsize=3)
    axes[0].set_title("A. Strict removal success")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")

    first = np.asarray([
        item["first_goal_physically_removed"]["safe_success_rate"] for item in cohorts
    ])
    second = np.asarray([
        item["second_goal_physically_removed"]["safe_success_rate"] for item in cohorts
    ])
    axes[1].bar(x - width / 2, first, width, color="#8058a5", label="First removed")
    axes[1].bar(x + width / 2, second, width, color="#4f9d69", label="Second removed")
    first_err = np.asarray([
        _interval_errors(
            value,
            item["first_goal_physically_removed"][
                "safe_success_hierarchical_bootstrap_95"
            ],
        )
        for value, item in zip(first, cohorts)
    ]).T
    second_err = np.asarray([
        _interval_errors(
            value,
            item["second_goal_physically_removed"][
                "safe_success_hierarchical_bootstrap_95"
            ],
        )
        for value, item in zip(second, cohorts)
    ]).T
    axes[1].errorbar(x - width / 2, first, yerr=first_err, fmt="none", ecolor="black", capsize=3)
    axes[1].errorbar(x + width / 2, second, yerr=second_err, fmt="none", ecolor="black", capsize=3)
    axes[1].set_title("B. Safe success by removed goal")
    axes[1].legend(frameon=False, fontsize=8, loc="upper center")

    violations = np.asarray([item["constraint_violation_rate"] for item in cohorts])
    axes[2].bar(x, violations, width=0.62, color=colors)
    violation_err = np.asarray([
        _interval_errors(
            value, item["constraint_violation_hierarchical_bootstrap_95"],
        )
        for value, item in zip(violations, cohorts)
    ]).T
    axes[2].errorbar(x, violations, yerr=violation_err, fmt="none", ecolor="black", capsize=3)
    axes[2].set_title("C. Constraint violations")

    for axis in axes:
        axis.set_xticks(x, labels, rotation=28, ha="right")
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Rate")
        axis.grid(axis="y", alpha=0.2, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Matched held-out physical-removal evaluation", fontsize=13)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    metadata = {
        "schema_version": 1,
        "protocol": payload["protocol"],
        "aggregate": str(aggregate_path),
        "aggregate_sha256": hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cohorts": [item["label"] for item in cohorts],
        "interval": "hierarchical training-seed/episode bootstrap",
        "outputs": [
            str(output.with_suffix(suffix))
            for suffix in (".md", ".csv", ".png", ".pdf")
        ],
    }
    _atomic_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        output.with_suffix(".metadata.json"),
    )


if __name__ == "__main__":
    main()
