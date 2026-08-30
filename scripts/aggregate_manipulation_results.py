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

from aggregate_visual_recovery import hierarchical_binary_interval
from evaluation_seed import validate_record_batch_seeds


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


def _paired_branch_effect(adaptive, baseline, keys, branch_value, rng) -> dict:
    branch_keys = [
        key for key in keys
        if int(adaptive[key]["first_goal_removed"] >= 0.5) == branch_value
    ]
    if not branch_keys:
        return {"paired_episodes": 0}
    if any(
        int(baseline[key]["first_goal_removed"] >= 0.5) != branch_value
        for key in branch_keys
    ):
        raise ValueError("paired methods disagree on first-goal-removed branch")
    raw = np.asarray([
        float(_metric_success_record(adaptive[key]))
        - float(_metric_success_record(baseline[key]))
        for key in branch_keys
    ])
    safe = np.asarray([
        float(
            _metric_success_record(adaptive[key])
            and adaptive[key].get("constraint_violated", 0.0) < 0.5
        )
        - float(
            _metric_success_record(baseline[key])
            and baseline[key].get("constraint_violated", 0.0) < 0.5
        )
        for key in branch_keys
    ])
    raw_samples = rng.choice(raw, size=(20000, len(raw)), replace=True).mean(axis=1)
    safe_samples = rng.choice(safe, size=(20000, len(safe)), replace=True).mean(axis=1)
    return {
        "paired_episodes": len(branch_keys),
        "success_rate_difference": float(raw.mean()),
        "paired_bootstrap_95": [
            float(np.quantile(raw_samples, 0.025)),
            float(np.quantile(raw_samples, 0.975)),
        ],
        "safe_success_rate_difference": float(safe.mean()),
        "safe_paired_bootstrap_95": [
            float(np.quantile(safe_samples, 0.025)),
            float(np.quantile(safe_samples, 0.975)),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--output", default="results/manipulation_ppo")
    parser.add_argument("--filename", default="aggregate.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rng = np.random.default_rng(20260828)
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
            record = json.loads(path.read_text(encoding="utf-8"))
            validate_record_batch_seeds(record, int(record["episodes"]))
            records.append(record)
            nominal_path = run_dir / "heldout_eval_nominal.json"
            if nominal_path.exists():
                nominal_record = json.loads(
                    nominal_path.read_text(encoding="utf-8")
                )
                validate_record_batch_seeds(
                    nominal_record, int(nominal_record["episodes"])
                )
                nominal_records.append(nominal_record)

    environments = []
    for experiment in config["experiments"]:
        env_id = experiment["env_id"]
        method = experiment.get("method", env_id)
        subset = [record for record in records if record.get("method", record["env_id"]) == method]
        successes = sum(record["successes"] for record in subset)
        episodes = sum(record["episodes"] for record in subset)
        success_trials = sum(record.get("success_trials", record["episodes"]) for record in subset)
        seed_rates = [record["success_rate"] for record in subset]
        seed_safe_rates = []
        for record in subset:
            seed_episodes = record.get("episode_records", [])
            if seed_episodes:
                seed_safe_rates.append(float(np.mean([
                    _metric_success_record(episode)
                    and episode.get("constraint_violated", 0.0) < 0.5
                    for episode in seed_episodes
                ])))
        episodes_raw = [
            episode for record in subset for episode in record.get("episode_records", [])
        ]
        seed_episode_groups = [
            record.get("episode_records", []) for record in subset
        ]
        safe_successes = sum(
            _metric_success_record(episode)
            and float(episode.get("constraint_violated", 0.0)) < 0.5
            for episode in episodes_raw
        )
        branch_success = {}
        for branch_key in ("first_goal_removed", "instruction_red_first"):
            if episodes_raw and all(branch_key in episode for episode in episodes_raw):
                branch_success[branch_key] = {}
                for branch_value in (0, 1):
                    branch = [
                        episode for episode in episodes_raw
                        if int(episode[branch_key] >= 0.5) == branch_value
                    ]
                    branch_raw_successes = sum(
                        _metric_success_record(episode) for episode in branch
                    )
                    branch_safe_successes = sum(
                        _metric_success_record(episode)
                        and episode.get("constraint_violated", 0.0) < 0.5
                        for episode in branch
                    )
                    branch_success[branch_key][str(branch_value)] = {
                        "episodes": len(branch),
                        "success_rate": branch_raw_successes / len(branch),
                        "success_wilson_95": _wilson(branch_raw_successes, len(branch)),
                        "constraint_violation_rate": float(np.mean([
                            episode.get("constraint_violated", 0.0) for episode in branch
                        ])),
                        "safe_success_rate": branch_safe_successes / len(branch),
                        "safe_success_wilson_95": _wilson(
                            branch_safe_successes, len(branch)
                        ),
                    }
        environment_result = {
            "env_id": env_id,
            "method": method,
            "seeds": len(subset),
            "episodes": episodes,
            "success_trials": success_trials,
            "successes": successes,
            "pooled_success_rate": successes / success_trials,
            "pooled_success_wilson_95": _wilson(successes, success_trials),
            "success_hierarchical_bootstrap_95": (
                hierarchical_binary_interval(
                    seed_episode_groups, _metric_success_record, rng
                ) if episodes_raw and all(seed_episode_groups) else None
            ),
            "seed_success_mean": float(np.mean(seed_rates)),
            "seed_success_std": float(np.std(seed_rates, ddof=1)) if len(seed_rates) > 1 else 0.0,
            "constraint_violation_rate": (
                float(np.mean([episode.get("constraint_violated", 0.0) for episode in episodes_raw]))
                if episodes_raw else None
            ),
            "safe_successes": safe_successes if episodes_raw else None,
            "pooled_safe_success_rate": (
                safe_successes / len(episodes_raw) if episodes_raw else None
            ),
            "pooled_safe_success_wilson_95": (
                _wilson(safe_successes, len(episodes_raw)) if episodes_raw else None
            ),
            "safe_success_hierarchical_bootstrap_95": (
                hierarchical_binary_interval(
                    seed_episode_groups,
                    lambda episode: _metric_success_record(episode)
                    and episode.get("constraint_violated", 0.0) < 0.5,
                    rng,
                ) if episodes_raw and all(seed_episode_groups) else None
            ),
            "seed_safe_success_mean": (
                float(np.mean(seed_safe_rates)) if seed_safe_rates else None
            ),
            "seed_safe_success_std": (
                float(np.std(seed_safe_rates, ddof=1))
                if len(seed_safe_rates) > 1 else (0.0 if seed_safe_rates else None)
            ),
            "mean_goals_completed": (
                float(np.mean([episode.get("goals_completed", 0.0) for episode in episodes_raw]))
                if episodes_raw else None
            ),
            "branch_success": branch_success,
            "seed_results": subset,
        }
        environments.append(environment_result)
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "protocol": "held-out deterministic state-policy evaluation",
        "environments": environments,
    }
    semantics = {record.get("benchmark_semantics") for record in records}
    if len(semantics) == 1 and None not in semantics:
        payload["benchmark_semantics"] = semantics.pop()
    if nominal_records:
        payload["nominal_condition"] = []
        for experiment in config["experiments"]:
            method = experiment.get("method", experiment["env_id"])
            subset = [record for record in nominal_records if record.get("method") == method]
            successes = sum(record["successes"] for record in subset)
            trials = sum(record["success_trials"] for record in subset)
            episodes_raw = [
                episode for record in subset for episode in record.get("episode_records", [])
            ]
            seed_episode_groups = [
                record.get("episode_records", []) for record in subset
            ]
            safe_successes = sum(
                _metric_success_record(episode)
                and episode.get("constraint_violated", 0.0) < 0.5
                for episode in episodes_raw
            )
            nominal_result = {
                "method": method,
                "seeds": len(subset),
                "episodes": trials,
                "successes": successes,
                "pooled_success_rate": successes / trials,
                "pooled_success_wilson_95": _wilson(successes, trials),
                "success_hierarchical_bootstrap_95": (
                    hierarchical_binary_interval(
                        seed_episode_groups, _metric_success_record, rng
                    ) if episodes_raw and all(seed_episode_groups) else None
                ),
                "constraint_violation_rate": (
                    float(np.mean([
                        episode.get("constraint_violated", 0.0) for episode in episodes_raw
                    ])) if episodes_raw else None
                ),
                "safe_successes": safe_successes if episodes_raw else None,
                "pooled_safe_success_rate": (
                    safe_successes / len(episodes_raw) if episodes_raw else None
                ),
                "pooled_safe_success_wilson_95": (
                    _wilson(safe_successes, len(episodes_raw)) if episodes_raw else None
                ),
                "safe_success_hierarchical_bootstrap_95": (
                    hierarchical_binary_interval(
                        seed_episode_groups,
                        lambda episode: _metric_success_record(episode)
                        and episode.get("constraint_violated", 0.0) < 0.5,
                        rng,
                    ) if episodes_raw and all(seed_episode_groups) else None
                ),
            }
            payload["nominal_condition"].append(nominal_result)
    # Recovery configs store per-episode records and use common held-out reset
    # seeds across methods. Report paired adaptive-policy effects in addition
    # to per-method Wilson intervals; stock-task configs remain unchanged.
    by_method = {
        result["method"]: result for result in environments
        if result.get("method")
    }
    adaptive_method = next(
        (
            method for method in by_method
            if "adaptive_ppo" in method and "no_intervention" not in method
        ),
        None,
    )
    if adaptive_method is not None:
        adaptive_records = {
            (record["training_seed"], index): episode
            for record in records if record.get("method") == adaptive_method
            for index, episode in enumerate(record.get("episode_records", []))
        }
        comparisons = []
        rng = np.random.default_rng(20260827)
        baselines = [
            method for method in by_method
            if "no_intervention_training_ppo" in method
            or "privileged_oracle_ppo" in method
        ]
        for baseline in baselines:
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
            safe_differences = np.asarray([
                float(
                    _metric_success_record(adaptive_records[key])
                    and adaptive_records[key].get("constraint_violated", 0.0) < 0.5
                )
                - float(
                    _metric_success_record(baseline_records[key])
                    and baseline_records[key].get("constraint_violated", 0.0) < 0.5
                )
                for key in keys
            ])
            samples = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
            safe_samples = rng.choice(
                safe_differences, size=(20000, len(safe_differences)), replace=True
            ).mean(axis=1)
            comparisons.append({
                "adaptive_method": adaptive_method,
                "adaptive_minus": baseline,
                "paired_episodes": len(keys),
                "success_rate_difference": float(differences.mean()),
                "paired_bootstrap_95": [
                    float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
                ],
                "safe_success_rate_difference": float(safe_differences.mean()),
                "safe_paired_bootstrap_95": [
                    float(np.quantile(safe_samples, 0.025)),
                    float(np.quantile(safe_samples, 0.975)),
                ],
                "first_goal_removed_branches": {
                    str(branch_value): _paired_branch_effect(
                        adaptive_records, baseline_records, keys, branch_value, rng
                    )
                    for branch_value in (0, 1)
                },
            })
        payload["paired_comparisons"] = comparisons
    _atomic_json(payload, root / args.filename)
    csv_path = root / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "environment", "training_seeds", "heldout_episodes", "successes",
            "pooled_success_rate", "wilson_95_low", "wilson_95_high",
            "seed_success_mean", "seed_success_std", "constraint_violation_rate",
            "safe_successes", "pooled_safe_success_rate", "safe_wilson_95_low",
            "safe_wilson_95_high", "seed_safe_success_mean", "seed_safe_success_std",
        ])
        for result in environments:
            safe_interval = result["pooled_safe_success_wilson_95"] or [None, None]
            writer.writerow([
                result["method"], result["env_id"], result["seeds"], result["episodes"], result["successes"],
                result["pooled_success_rate"], *result["pooled_success_wilson_95"],
                result["seed_success_mean"], result["seed_success_std"],
                result["constraint_violation_rate"], result["safe_successes"],
                result["pooled_safe_success_rate"], *safe_interval,
                result["seed_safe_success_mean"], result["seed_safe_success_std"],
            ])
    os.replace(temporary, csv_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
