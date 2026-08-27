#!/usr/bin/env python3
"""Create paper-ready manipulation learning curves from immutable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    aggregate_path = root / "aggregate.json"
    aggregate = (
        json.loads(aggregate_path.read_text(encoding="utf-8"))
        if aggregate_path.exists() else None
    )

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    })
    experiments = config["experiments"]
    figure, axes = plt.subplots(1, len(experiments), figsize=(4.0 * len(experiments), 3.0))
    axes = np.atleast_1d(axes)
    for axis, experiment in zip(axes, experiments, strict=True):
        env_id = experiment["env_id"]
        curves = []
        for seed in config["seeds"]:
            path = root / env_id / f"seed_{seed}" / "metrics.jsonl"
            records = _records(path)
            x = np.asarray([record["global_step"] for record in records], dtype=float) / 1e6
            y = np.asarray([
                record["eval"].get(
                    "success_once", record["eval"].get("success_at_end", np.nan)
                )
                for record in records
            ], dtype=float)
            axis.plot(x, y, alpha=0.28, linewidth=1.0, label=f"seed {seed}")
            curves.append((x, y))
        common_x = curves[0][0]
        if not all(np.array_equal(common_x, x) for x, _ in curves[1:]):
            raise ValueError(f"evaluation steps differ across {env_id} seeds")
        values = np.stack([y for _, y in curves])
        axis.plot(common_x, np.nanmean(values, axis=0), color="black", linewidth=2, label="seed mean")
        if aggregate is not None:
            result = next(item for item in aggregate["environments"] if item["env_id"] == env_id)
            heldout = result["pooled_success_rate"]
            low, high = result["pooled_success_wilson_95"]
            axis.axhline(heldout, color="#c23b22", linestyle="--", linewidth=1.5, label="held-out pooled")
            axis.fill_between([common_x.min(), common_x.max()], low, high, color="#c23b22", alpha=0.10)
        axis.set_title(env_id.replace("-v1", ""))
        axis.set_xlabel("Environment transitions (millions)")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Success rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=len(by_label), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    for suffix in ("png", "pdf"):
        path = root / f"learning_curves.{suffix}"
        figure.savefig(path, bbox_inches="tight")
        print(path)
    plt.close(figure)


if __name__ == "__main__":
    main()
