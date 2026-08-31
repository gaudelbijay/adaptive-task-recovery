#!/usr/bin/env python3
"""Aggregate repeated-seed REBOOT leave-one-object-out evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHODS = ("static_mlp", "moment_mlp", "unstructured_gru", "causal_dynamics_gru")


def object_bootstrap(values: np.ndarray, seed: int, samples: int) -> list[float]:
    """Bootstrap held-out objects after averaging optimizer seeds per object."""
    per_object = values.mean(axis=0)
    rng = np.random.default_rng(seed)
    draws = np.asarray([
        rng.choice(per_object, size=len(per_object), replace=True).mean()
        for _ in range(samples)
    ])
    return [float(x) for x in np.quantile(draws, (0.025, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    runs = [json.loads(path.read_text()) for path in args.inputs]
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("optimizer seeds must be unique")
    object_order = [[fold["test_object"] for fold in run["folds"]] for run in runs]
    if any(order != object_order[0] for order in object_order[1:]):
        raise ValueError("held-out object order differs across runs")

    aggregate = {}
    for method in METHODS:
        values = np.asarray([[fold[method]["auroc"] for fold in run["folds"]] for run in runs])
        aggregate[method] = {
            "macro_auroc_mean": float(values.mean()),
            "optimizer_seed_means": [float(x) for x in values.mean(axis=1)],
        }
    comparisons = {}
    candidate = np.asarray([
        [fold["causal_dynamics_gru"]["auroc"] for fold in run["folds"]]
        for run in runs
    ])
    for baseline in METHODS[:-1]:
        baseline_values = np.asarray([
            [fold[baseline]["auroc"] for fold in run["folds"]]
            for run in runs
        ])
        difference = candidate - baseline_values
        comparisons[f"causal_vs_{baseline}"] = {
            "macro_auroc_difference": float(difference.mean()),
            "optimizer_seed_differences": [float(x) for x in difference.mean(axis=1)],
            "object_bootstrap_95": object_bootstrap(
                difference, args.seed, args.bootstrap_samples,
            ),
        }
    payload = {
        "schema_version": 1,
        "protocol": runs[0]["protocol"],
        "claim_boundary": runs[0]["claim_boundary"],
        "optimizer_seeds": sorted(int(run["seed"]) for run in runs),
        "held_out_objects": object_order[0],
        "episodes": int(sum(fold["static_mlp"]["episodes"] for fold in runs[0]["folds"])),
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
