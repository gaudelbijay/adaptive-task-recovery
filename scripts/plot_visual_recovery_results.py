#!/usr/bin/env python3
"""Render the final held-out V3 visual-recovery comparison figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = {
    "direct_rgb_ppo": "Direct\nRGB PPO",
    "asymmetric_rgb_ppo": "Asym.\nRGB PPO",
    "asymmetric_temporal_rgb_ppo": "Asym. +\ntemporal",
    "pose_aux_dagger_visual_ppo": "DAgger\nno SSL",
    "pose_aux_temporal_dagger_visual_ppo": "DAgger +\ntemporal",
    "learned_progress_temporal_dagger_visual_ppo": "DAgger +\nprogress",
    "learned_progress_adaptive_visual_ppo": "Adaptive +\nprogress",
    "pose_aux_dagger_adaptive_visual_ppo": "Adaptive\nno SSL",
    "pose_aux_temporal_dagger_adaptive_visual_ppo": "Adaptive +\ntemporal",
}


def interval_errors(value, interval):
    low, high = map(float, interval)
    value = float(value)
    return [max(0.0, value - low), max(0.0, high - value)]


def validate(payload):
    if payload.get("benchmark_semantics") != "event_reward_intervention_target_only_v3":
        raise ValueError("figure input does not use V3 event-reward semantics")
    if payload.get("required_missing"):
        raise ValueError("figure input is missing required primary aggregates")
    rows = payload.get("candidates", [])
    if not rows:
        raise ValueError("figure input contains no visual candidates")
    required = {
        "name", "nominal_success_rate",
        "nominal_success_hierarchical_bootstrap_95", "safe_success_rate",
        "safe_success_hierarchical_bootstrap_95", "constraint_violation_rate",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"figure candidate lacks fields: {sorted(missing)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison", default="results/final_visual_comparison/comparison.json"
    )
    parser.add_argument(
        "--output-prefix", default="media/results/v3_visual_recovery_comparison"
    )
    args = parser.parse_args()
    payload = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    rows = validate(payload)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [LABELS.get(row["name"], row["name"].replace("_", "\n")) for row in rows]
    x = np.arange(len(rows))
    colors = [
        "#6b7280" if "direct" in row["name"] else
        "#7c3aed" if "temporal" in row["name"] else
        "#0f766e" if "adaptive" in row["name"] else "#2563eb"
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    panels = [
        (
            "Nominal ordered-task success",
            "nominal_success_rate", "nominal_success_hierarchical_bootstrap_95",
        ),
        (
            "Forced-sweeper-condition safe success",
            "safe_success_rate", "safe_success_hierarchical_bootstrap_95",
        ),
    ]
    for axis, (title, value_key, interval_key) in zip(axes[:2], panels):
        values = np.asarray([row[value_key] for row in rows], dtype=float)
        errors = np.asarray([
            interval_errors(value, row[interval_key])
            for value, row in zip(values, rows)
        ]).T
        axis.bar(x, values * 100, color=colors, edgecolor="black", linewidth=0.5)
        axis.errorbar(
            x, values * 100, yerr=errors * 100, fmt="none", ecolor="black",
            elinewidth=1.2, capsize=3,
        )
        axis.set_title(title)
        axis.set_ylabel("Success (%)")
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x, names, rotation=0, fontsize=8)

    state = payload["reference"]
    axes[1].axhline(
        float(state["safe_success_rate"]) * 100, color="#b91c1c",
        linestyle="--", linewidth=1.5, label="State reference",
    )
    axes[1].legend(frameon=False, fontsize=8)

    violations = np.asarray([row["constraint_violation_rate"] for row in rows]) * 100
    axes[2].bar(x, violations, color=colors, edgecolor="black", linewidth=0.5)
    axes[2].axhline(
        float(state["constraint_violation_rate"]) * 100, color="#b91c1c",
        linestyle="--", linewidth=1.5, label="State reference maximum",
    )
    axes[2].set_title("Forced-sweeper-condition violations")
    axes[2].set_ylabel("Episodes with violation (%)")
    axes[2].set_ylim(bottom=0)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_xticks(x, names, rotation=0, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Held-out V3 control/sweeper condition — 3 training seeds × 256 episodes/condition",
        fontsize=12,
    )
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {prefix.with_suffix('.png')} and {prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
