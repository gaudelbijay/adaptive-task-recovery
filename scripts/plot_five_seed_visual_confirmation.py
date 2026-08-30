#!/usr/bin/env python3
"""Render the strict five-seed held-out visual/state confirmation figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def interval_errors(value, interval):
    return np.asarray([[value - interval[0]], [interval[1] - value]])


def validate(payload):
    if payload.get("protocol") != "five-seed confirmatory held-out visual/state comparison":
        raise ValueError("five-seed figure input has the wrong protocol")
    if payload.get("benchmark_semantics") != "event_reward_intervention_target_only_v3":
        raise ValueError("five-seed figure input has the wrong semantics")
    if payload.get("no_seed_discarded") is not True:
        raise ValueError("five-seed figure refuses selectively filtered seeds")
    if len(payload.get("all_training_seeds", [])) != 5:
        raise ValueError("five-seed figure requires exactly five training seeds")
    if int(payload.get("heldout_episodes_per_condition", -1)) != 1280:
        raise ValueError("five-seed figure requires 1,280 episodes per condition")
    if not payload.get("visual_training_source_sha256"):
        raise ValueError("five-seed figure lacks visual training provenance")
    if not payload.get("visual_evaluation_source_sha256"):
        raise ValueError("five-seed figure lacks visual evaluation provenance")
    visual = payload.get("visual", {})
    for condition in ("nominal", "intervention"):
        result = visual.get(condition, {})
        if int(result.get("seeds", -1)) != 5 or int(result.get("episodes", -1)) != 1280:
            raise ValueError(f"five-seed figure has incomplete {condition} results")
    state = payload.get("state_forced_intervention", {})
    if int(state.get("seeds", -1)) != 5 or int(state.get("episodes", -1)) != 1280:
        raise ValueError("five-seed figure has incomplete state results")
    return visual["nominal"], visual["intervention"], state


def render(payload, output):
    import matplotlib.pyplot as plt

    nominal, visual, state = validate(payload)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)

    competence = [nominal["success_rate"]]
    axes[0].bar([0], competence, color="#2f6f9f", width=0.58)
    axes[0].errorbar(
        [0], competence,
        yerr=interval_errors(
            competence[0], nominal["success_hierarchical_bootstrap_95"],
        ),
        fmt="none", ecolor="black", capsize=5,
    )
    axes[0].axhline(0.70, color="#555555", linestyle="--", linewidth=1.2)
    axes[0].set_xticks([0], ["Visual"])
    axes[0].set_title("Nominal retention")
    axes[0].set_ylabel("Held-out success")

    labels = ["Visual", "State"]
    raw = [visual["success_rate"], state["success_rate"]]
    safe = [visual["safe_success_rate"], state["safe_success_rate"]]
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, raw, width, label="Raw", color="#4c78a8")
    axes[1].bar(x + width / 2, safe, width, label="Safe", color="#59a14f")
    for index, (value, interval) in enumerate(zip(
        safe,
        [visual["safe_success_hierarchical_bootstrap_95"], state["safe_success_hierarchical_bootstrap_95"]],
    )):
        axes[1].errorbar(
            [x[index] + width / 2], [value], yerr=interval_errors(value, interval),
            fmt="none", ecolor="black", capsize=4,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Forced intervention")
    axes[1].legend(frameon=False, loc="upper right")

    violations = [visual["constraint_violation_rate"], state["constraint_violation_rate"]]
    axes[2].bar(x, violations, color=["#e15759", "#bab0ac"], width=0.58)
    axes[2].set_xticks(x, labels)
    axes[2].set_title("Safety violations")
    axes[2].set_ylabel("Episode rate")

    for axis in axes:
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Non-teleport RGB manipulation recovery — five training seeds, 1,280 episodes/condition",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="results/final_visual_comparison/five_seed_confirmation.json",
    )
    parser.add_argument(
        "--output", default="media/results/v3_visual_recovery_five_seed.png",
    )
    args = parser.parse_args()
    render(
        json.loads(Path(args.input).read_text(encoding="utf-8")),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
