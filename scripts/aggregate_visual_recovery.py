#!/usr/bin/env python3
"""Aggregate visual-policy evaluations with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np


def wilson(successes, trials, z=1.959963984540054):
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def succeeded(record):
    for key in ("success_once", "success_at_end", "success"):
        if key in record:
            return record[key] >= 0.5
    raise KeyError("episode has no success field")


def paired_effect(left, right, rng):
    if len(left) != len(right):
        raise ValueError("paired methods have unequal episode counts")
    raw = np.asarray([float(succeeded(a)) - float(succeeded(b)) for a, b in zip(left, right)])
    safe = np.asarray([
        float(succeeded(a) and a.get("constraint_violated", 0.0) < 0.5)
        - float(succeeded(b) and b.get("constraint_violated", 0.0) < 0.5)
        for a, b in zip(left, right)
    ])
    indices = rng.integers(0, len(raw), size=(20000, len(raw)))
    raw_samples = raw[indices].mean(1)
    safe_samples = safe[indices].mean(1)
    return {
        "paired_episodes": len(raw),
        "success_rate_difference": float(raw.mean()),
        "paired_bootstrap_95": [float(np.quantile(raw_samples, 0.025)), float(np.quantile(raw_samples, 0.975))],
        "safe_success_rate_difference": float(safe.mean()),
        "safe_paired_bootstrap_95": [float(np.quantile(safe_samples, 0.025)), float(np.quantile(safe_samples, 0.975))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_ppo_gate_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "benchmark_semantics": "intervention_target_only_v2",
        "conditions": {},
    }
    rng = np.random.default_rng(20260827)
    for condition in ("nominal", "intervention"):
        filename = f"heldout_eval_{condition}.json"
        by_method = {}
        for experiment in config["experiments"]:
            method = experiment["method"]
            records = []
            for seed in config["seeds"]:
                path = root / method / f"seed_{seed}" / filename
                if not path.exists():
                    raise FileNotFoundError(path)
                records.append(json.loads(path.read_text(encoding="utf-8")))
            episodes = [episode for record in records for episode in record["episode_records"]]
            successes = sum(succeeded(episode) for episode in episodes)
            safe = sum(succeeded(episode) and episode.get("constraint_violated", 0.0) < 0.5 for episode in episodes)
            by_method[method] = {
                "method": method, "seeds": len(records), "episodes": len(episodes),
                "successes": successes, "success_rate": successes / len(episodes),
                "success_wilson_95": wilson(successes, len(episodes)),
                "safe_successes": safe, "safe_success_rate": safe / len(episodes),
                "safe_success_wilson_95": wilson(safe, len(episodes)),
                "constraint_violation_rate": float(np.mean([episode.get("constraint_violated", 0.0) for episode in episodes])),
                "mean_goals_completed": float(np.mean([episode.get("goals_completed", 0.0) for episode in episodes])),
                "seed_success_rates": [record["success_rate"] for record in records],
                "checkpoint_global_steps": [record["checkpoint_global_step"] for record in records],
            }
            by_method[method]["_episodes"] = episodes
        comparisons = []
        for left, right in itertools.combinations(by_method, 2):
            result = {"left": left, "right": right}
            result.update(paired_effect(by_method[left]["_episodes"], by_method[right]["_episodes"], rng))
            comparisons.append(result)
        for result in by_method.values():
            result.pop("_episodes")
        payload["conditions"][condition] = {
            "methods": list(by_method.values()), "paired_comparisons": comparisons,
        }
    path = root / "aggregate.json"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
