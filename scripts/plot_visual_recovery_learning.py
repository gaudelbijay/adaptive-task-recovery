#!/usr/bin/env python3
"""Plot seed-aware visual-recovery learning curves from immutable JSONL logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    ("success_once", "Full-task success", (0.0, 1.02)),
    ("fail_once", "Constraint violation", (0.0, 1.02)),
    ("goals_completed", "Mean goals completed", (0.0, 2.02)),
    ("visual_progress_bit_accuracy", "Progress-bit accuracy", (0.0, 1.02)),
)


def method_label(experiment):
    if experiment.get("bc_teacher_checkpoint"):
        parts = ["DAgger"]
    else:
        parts = ["RGB PPO"]
    if experiment.get("asymmetric_critic"):
        parts.append("asymmetric critic")
    if experiment.get("privileged_aux_coefficient", 0.0):
        parts.append("pose auxiliary")
    if experiment.get("temporal_ssl_coefficient", 0.0):
        parts.append("temporal SSL")
    if experiment.get("actor_learned_goal_progress"):
        parts.append("progress head")
    return " + ".join(parts)


def load_method(config_path: Path, output_root: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = output_root / config["name"]
    for experiment in config["experiments"]:
        seed_curves = []
        for seed in config["seeds"]:
            path = root / experiment["method"] / f"seed_{seed}" / "metrics.jsonl"
            if not path.exists():
                continue
            evaluations = []
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if "eval" in record:
                        evaluations.append(record)
            if evaluations:
                seed_curves.append((seed, evaluations))
        if seed_curves:
            yield (
                config["name"], experiment["method"],
                method_label(experiment), seed_curves,
            )


def aggregate_at_shared_steps(seed_curves, metric):
    per_seed = []
    for seed, records in seed_curves:
        values = {
            int(record["global_step"]): float(record["eval"][metric])
            for record in records if metric in record["eval"]
        }
        if values:
            per_seed.append((seed, values))
    if not per_seed:
        return None
    shared = sorted(set.intersection(*(set(values) for _, values in per_seed)))
    if not shared:
        return None
    matrix = np.asarray([[values[step] for step in shared] for _, values in per_seed])
    return np.asarray(shared) / 1_000_000.0, matrix, [seed for seed, _ in per_seed]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--output-root", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--figure-stem", default="media/results/visual-recovery-v3-learning"
    )
    args = parser.parse_args()

    methods = []
    for path in args.config:
        methods.extend(load_method(Path(path), Path(args.output_root)))
    if not methods:
        raise FileNotFoundError("no visual recovery metrics found")

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), sharex=True)
    colors = plt.get_cmap("tab10")
    for method_index, (experiment_name, method, label, seed_curves) in enumerate(methods):
        color = colors(method_index % 10)
        for axis, (metric, title, limits) in zip(axes.flat, METRICS):
            aggregate = aggregate_at_shared_steps(seed_curves, metric)
            if aggregate is None:
                continue
            steps, matrix, _ = aggregate
            for values in matrix:
                axis.plot(steps, values, color=color, alpha=0.18, linewidth=0.8)
            mean = matrix.mean(axis=0)
            axis.plot(steps, mean, color=color, linewidth=2.0, label=label)
            if len(matrix) > 1:
                std = matrix.std(axis=0, ddof=1)
                axis.fill_between(
                    steps, np.clip(mean - std, limits[0], limits[1]),
                    np.clip(mean + std, limits[0], limits[1]),
                    color=color, alpha=0.10, linewidth=0,
                )
            axis.set_title(title)
            axis.set_ylim(*limits)
            axis.grid(alpha=0.25)

    axes[0, 0].axhline(0.70, color="black", linestyle="--", linewidth=1.0)
    axes[0, 0].text(0.99, 0.715, "V1 gate", ha="right", va="bottom",
                    transform=axes[0, 0].get_yaxis_transform(), fontsize=8)
    for axis in axes[1]:
        axis.set_xlabel("PPO environment steps (millions)")
    axes[0, 0].set_ylabel("Rate")
    axes[0, 1].set_ylabel("Rate")
    axes[1, 0].set_ylabel("Goals")
    axes[1, 1].set_ylabel("Accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                  bbox_to_anchor=(0.5, -0.01))
    figure.suptitle("V3 visual recovery training-stream diagnostics (not held-out)")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))

    stem = Path(args.figure_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps({
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
        "methods": [method for _, method, _, _ in methods],
        "uncertainty": "mean +/- sample standard deviation across training seeds",
        "claim_boundary": "training-stream diagnostics; not held-out evaluation",
    }, indent=2))


if __name__ == "__main__":
    main()
