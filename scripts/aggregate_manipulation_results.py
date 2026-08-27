#!/usr/bin/env python3
"""Aggregate immutable held-out manipulation evaluations across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    records = []
    for experiment in config["experiments"]:
        env_id = experiment["env_id"]
        for seed in config["seeds"]:
            path = root / env_id / f"seed_{seed}" / "heldout_eval.json"
            if not path.exists():
                raise FileNotFoundError(f"missing held-out result: {path}")
            records.append(json.loads(path.read_text(encoding="utf-8")))

    environments = []
    for experiment in config["experiments"]:
        env_id = experiment["env_id"]
        subset = [record for record in records if record["env_id"] == env_id]
        successes = sum(record["successes"] for record in subset)
        episodes = sum(record["episodes"] for record in subset)
        success_trials = sum(record.get("success_trials", record["episodes"]) for record in subset)
        seed_rates = [record["success_rate"] for record in subset]
        environments.append({
            "env_id": env_id,
            "seeds": len(subset),
            "episodes": episodes,
            "success_trials": success_trials,
            "successes": successes,
            "pooled_success_rate": successes / success_trials,
            "pooled_success_wilson_95": _wilson(successes, success_trials),
            "seed_success_mean": float(np.mean(seed_rates)),
            "seed_success_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
            "seed_results": subset,
        })
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "protocol": "held-out deterministic state-policy evaluation",
        "environments": environments,
    }
    _atomic_json(payload, root / "aggregate.json")
    csv_path = root / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "environment", "training_seeds", "heldout_episodes", "successes",
            "pooled_success_rate", "wilson_95_low", "wilson_95_high",
            "seed_success_mean", "seed_success_std",
        ])
        for result in environments:
            writer.writerow([
                result["env_id"], result["seeds"], result["episodes"], result["successes"],
                result["pooled_success_rate"], *result["pooled_success_wilson_95"],
                result["seed_success_mean"], result["seed_success_std"],
            ])
    os.replace(temporary, csv_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
