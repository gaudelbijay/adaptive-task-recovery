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


def _metric_success_record(record: dict) -> bool:
    for key in ("success_once", "success_at_end", "success"):
        if key in record:
            return float(record[key]) >= 0.5
    raise KeyError("episode record has no success metric")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    records = []
    nominal_records = []
    for experiment in config["experiments"]:
        env_id = experiment["env_id"]
        method = experiment.get("method", env_id)
        for seed in config["seeds"]:
            run_dir = root / method / f"seed_{seed}"
            path = run_dir / "heldout_eval_intervention.json"
            if not path.exists():
                path = run_dir / "heldout_eval.json"
            if not path.exists():
                raise FileNotFoundError(f"missing held-out result: {path}")
            records.append(json.loads(path.read_text(encoding="utf-8")))
            nominal_path = run_dir / "heldout_eval_nominal.json"
            if nominal_path.exists():
                nominal_records.append(json.loads(nominal_path.read_text(encoding="utf-8")))

    environments = []
    for experiment in config["experiments"]:
        env_id = experiment["env_id"]
        method = experiment.get("method", env_id)
        subset = [record for record in records if record.get("method", record["env_id"]) == method]
        successes = sum(record["successes"] for record in subset)
        episodes = sum(record["episodes"] for record in subset)
        success_trials = sum(record.get("success_trials", record["episodes"]) for record in subset)
        seed_rates = [record["success_rate"] for record in subset]
        episodes_raw = [
            episode for record in subset for episode in record.get("episode_records", [])
        ]
        branch_success = {}
        for branch_key in ("first_goal_removed", "instruction_red_first"):
            if episodes_raw and all(branch_key in episode for episode in episodes_raw):
                branch_success[branch_key] = {}
                for branch_value in (0, 1):
                    branch = [
                        episode for episode in episodes_raw
                        if int(episode[branch_key] >= 0.5) == branch_value
                    ]
                    branch_success[branch_key][str(branch_value)] = {
                        "episodes": len(branch),
                        "success_rate": float(np.mean([
                            _metric_success_record(episode) for episode in branch
                        ])),
                    }
        environments.append({
            "env_id": env_id,
            "method": method,
            "seeds": len(subset),
            "episodes": episodes,
            "success_trials": success_trials,
            "successes": successes,
            "pooled_success_rate": successes / success_trials,
            "pooled_success_wilson_95": _wilson(successes, success_trials),
            "seed_success_mean": float(np.mean(seed_rates)),
            "seed_success_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
            "constraint_violation_rate": (
                float(np.mean([episode.get("constraint_violated", 0.0) for episode in episodes_raw]))
                if episodes_raw else None
            ),
            "mean_goals_completed": (
                float(np.mean([episode.get("goals_completed", 0.0) for episode in episodes_raw]))
                if episodes_raw else None
            ),
            "branch_success": branch_success,
            "seed_results": subset,
        })
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "protocol": "held-out deterministic state-policy evaluation",
        "environments": environments,
    }
    if nominal_records:
        payload["nominal_condition"] = []
        for experiment in config["experiments"]:
            method = experiment.get("method", experiment["env_id"])
            subset = [record for record in nominal_records if record.get("method") == method]
            successes = sum(record["successes"] for record in subset)
            trials = sum(record["success_trials"] for record in subset)
            payload["nominal_condition"].append({
                "method": method,
                "seeds": len(subset),
                "episodes": trials,
                "successes": successes,
                "pooled_success_rate": successes / trials,
                "pooled_success_wilson_95": _wilson(successes, trials),
            })
    # Recovery configs store per-episode records and use common held-out reset
    # seeds across methods. Report paired adaptive-policy effects in addition
    # to per-method Wilson intervals; stock-task configs remain unchanged.
    by_method = {
        result["method"]: result for result in environments
        if result.get("method")
    }
    if "adaptive_ppo" in by_method:
        adaptive_records = {
            (record["training_seed"], index): episode
            for record in records if record.get("method") == "adaptive_ppo"
            for index, episode in enumerate(record.get("episode_records", []))
        }
        comparisons = []
        rng = np.random.default_rng(20260827)
        for baseline in ("no_intervention_training_ppo", "privileged_oracle_ppo"):
            baseline_records = {
                (record["training_seed"], index): episode
                for record in records if record.get("method") == baseline
                for index, episode in enumerate(record.get("episode_records", []))
            }
            keys = sorted(set(adaptive_records) & set(baseline_records))
            if not keys:
                continue
            differences = np.asarray([
                float(_metric_success_record(adaptive_records[key]))
                - float(_metric_success_record(baseline_records[key]))
                for key in keys
            ])
            samples = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
            comparisons.append({
                "adaptive_minus": baseline,
                "paired_episodes": len(keys),
                "success_rate_difference": float(differences.mean()),
                "paired_bootstrap_95": [
                    float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
                ],
            })
        payload["paired_comparisons"] = comparisons
    _atomic_json(payload, root / "aggregate.json")
    csv_path = root / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "environment", "training_seeds", "heldout_episodes", "successes",
            "pooled_success_rate", "wilson_95_low", "wilson_95_high",
            "seed_success_mean", "seed_success_std",
        ])
        for result in environments:
            writer.writerow([
                result["method"], result["env_id"], result["seeds"], result["episodes"], result["successes"],
                result["pooled_success_rate"], *result["pooled_success_wilson_95"],
                result["seed_success_mean"], result["seed_success_std"],
            ])
    os.replace(temporary, csv_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
