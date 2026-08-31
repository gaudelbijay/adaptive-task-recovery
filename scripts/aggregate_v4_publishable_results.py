#!/usr/bin/env python3
"""Aggregate the frozen V4 controller, baseline, and renderer-OOD evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CONDITIONS = (
    "nominal", "ejection", "permanent_block", "temporary_block",
    "reverse_ejection",
)


def wilson(successes: int, episodes: int, z: float = 1.959963984540054):
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = z * math.sqrt(
        rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)
    ) / denominator
    return [center - radius, center + radius]


def load_rows(folder: Path, *, baseline: bool = False):
    rows = []
    for path in sorted(folder.glob("*.json")):
        row = json.loads(path.read_text())
        if baseline and row.get("progress_source") != "normal":
            continue
        rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for condition in (*CONDITIONS, "all"):
        selected = rows if condition == "all" else [
            row for row in rows if row["condition"] == condition
        ]
        successes = sum(int(row["successes"]) for row in selected)
        violations = sum(int(row["violations"]) for row in selected)
        episodes = sum(int(row["episodes"]) for row in selected)
        if not episodes:
            continue
        result[condition] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "success_wilson_95": wilson(successes, episodes),
            "violations": violations,
            "violation_rate": violations / episodes,
            "violation_wilson_95": wilson(violations, episodes),
        }
    return result


def compare(controller, baseline):
    comparison = {}
    for condition in sorted(controller.keys() & baseline.keys()):
        c, b = controller[condition], baseline[condition]
        comparison[condition] = {
            "success_rate_difference": c["success_rate"] - b["success_rate"],
            "success_difference_newcombe_95": [
                c["success_wilson_95"][0] - b["success_wilson_95"][1],
                c["success_wilson_95"][1] - b["success_wilson_95"][0],
            ],
            "violation_rate_reduction": b["violation_rate"] - c["violation_rate"],
            "violation_reduction_newcombe_95": [
                b["violation_wilson_95"][0] - c["violation_wilson_95"][1],
                b["violation_wilson_95"][1] - c["violation_wilson_95"][0],
            ],
        }
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller", default="results/v4_temporal_controller_v28_confirmatory"
    )
    parser.add_argument(
        "--baseline", default="results/v19_on_v4_v28_confirmatory"
    )
    parser.add_argument(
        "--controller-ood", default="results/v4_temporal_controller_v27_ood"
    )
    parser.add_argument("--baseline-ood", default="results/v19_on_v4_ood")
    parser.add_argument("--output", default="results/v4_publishable_summary.json")
    args = parser.parse_args()

    controller = summarize(load_rows(Path(args.controller)))
    baseline = summarize(load_rows(Path(args.baseline), baseline=True))
    profiles = {}
    pooled_controller_ood = []
    pooled_baseline_ood = []
    controller_ood = Path(args.controller_ood)
    baseline_ood = Path(args.baseline_ood)
    if controller_ood.exists() and baseline_ood.exists():
        names = sorted(
            {path.name for path in controller_ood.iterdir() if path.is_dir()}
            & {path.name for path in baseline_ood.iterdir() if path.is_dir()}
        )
        for name in names:
            controller_rows = load_rows(controller_ood / name)
            baseline_rows = load_rows(baseline_ood / name, baseline=True)
            pooled_controller_ood.extend(controller_rows)
            pooled_baseline_ood.extend(baseline_rows)
            profiles[name] = {
                "controller": summarize(controller_rows),
                "baseline": summarize(baseline_rows),
            }
    payload = {
        "schema_version": 1,
        "controller": controller,
        "baseline": baseline,
        "comparison": compare(controller, baseline),
        "absolute_success_gain": (
            controller["all"]["success_rate"] - baseline["all"]["success_rate"]
        ),
        "absolute_violation_reduction": (
            baseline["all"]["violation_rate"] - controller["all"]["violation_rate"]
        ),
        "renderer_ood": profiles,
        "renderer_ood_pooled": (
            {
                "controller": summarize(pooled_controller_ood),
                "baseline": summarize(pooled_baseline_ood),
                "comparison": compare(
                    summarize(pooled_controller_ood),
                    summarize(pooled_baseline_ood),
                ),
            }
            if pooled_controller_ood and pooled_baseline_ood else {}
        ),
        "claim_boundary": (
            "Wilson intervals pool environment episodes. The controller uses "
            "explicit object-state observations for routing/specialists and RGB "
            "for its retained V19 branch; this is not a restricted-RGB result."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
