#!/usr/bin/env python3
"""Aggregate held-out task-semantic probes across trained visual seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_representation_probes import seed_bootstrap_interval


PROTOCOL = "held-out linear goal-resolution probe; labels unavailable to actor"
AGGREGATE_PROTOCOL = (
    "held-out linear goal-resolution probe aggregated over training seeds"
)
TARGETS = ["red_goal_resolved", "blue_goal_resolved"]
METRICS = {
    "r2": ("r2_variance_weighted", "learned_minus_random_r2"),
    "balanced_accuracy": (
        "macro_balanced_accuracy", "learned_minus_random_balanced_accuracy",
    ),
    "roc_auc": ("macro_roc_auc", "learned_minus_random_roc_auc"),
}


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def expected_observation_contract(experiment):
    if experiment.get("actor_learned_goal_progress", False):
        return "rgb_robot_proprio_instruction_visual_progress_v4"
    if experiment.get("actor_goal_progress", False):
        return "rgb_robot_proprio_instruction_progress_v3"
    if experiment.get("actor_tcp_pose", False):
        return "rgb_qpos_qvel_tcp_instruction_v2"
    return "rgb_qpos_qvel_instruction_v1"


def validate_record(record, experiment, seed, config):
    expected = {
        "protocol": PROTOCOL,
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "observation_contract": expected_observation_contract(experiment),
        "method": experiment["method"],
        "training_seed": seed,
        "train_samples": int(config.get("representation_probe_samples_per_split", 8192)),
        "test_samples": int(config.get("representation_probe_samples_per_split", 8192)),
        "ridge_regularization": float(config.get("representation_probe_ridge", 1.0)),
        "targets": TARGETS,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"task probe {key} mismatch: expected {value!r}, "
                f"observed {record.get(key)!r}"
            )
    if int(record.get("checkpoint_global_step", -1)) < 0:
        raise ValueError("task probe lacks checkpoint provenance")
    if not record.get("training_source_sha256", {}).get("trainer"):
        raise ValueError("task probe lacks training-source provenance")
    if not record.get("probe_source_sha256", {}).get("probe"):
        raise ValueError("task probe lacks probe-source provenance")
    dataset = record.get("probe_dataset", {})
    expected_behavior = config.get("representation_probe_behavior_checkpoint")
    if not expected_behavior:
        raise ValueError("task probe config lacks frozen behavior checkpoint")
    expected_dataset = {
        "protocol": "frozen seed-matched behavior policy; identical pixels across methods",
        "behavior_checkpoint": str(expected_behavior).format(seed=seed),
        "train_seed": 93000000 + seed * 10,
        "test_seed": 93000000 + seed * 10 + 1,
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise ValueError(f"task probe dataset {key} mismatch")
    for key in ("train_sha256", "test_sha256"):
        digest = dataset.get(key, "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"task probe dataset {key} is not SHA-256")
    for metric_name, (metric_key, delta_key) in METRICS.items():
        learned = float(record["learned_encoder"][metric_key])
        random = float(record["random_encoder"][metric_key])
        delta = float(record[delta_key])
        if not np.isfinite([learned, random, delta]).all():
            raise ValueError(f"task probe {metric_name} is non-finite")
        if not np.isclose(delta, learned - random, rtol=0.0, atol=1e-12):
            raise ValueError(f"task probe {metric_name} arithmetic mismatch")
    for encoder in ("learned_encoder", "random_encoder"):
        prevalence = record[encoder]["per_target_positive_prevalence"]
        if len(prevalence) != len(TARGETS) or not all(0 < float(x) < 1 for x in prevalence):
            raise ValueError("task probe target lacks both held-out classes")
    return record


def summarize(records, rng):
    summary = {}
    for name, (metric_key, delta_key) in METRICS.items():
        learned = np.asarray([
            record["learned_encoder"][metric_key] for record in records
        ], dtype=float)
        random = np.asarray([
            record["random_encoder"][metric_key] for record in records
        ], dtype=float)
        delta = np.asarray([record[delta_key] for record in records], dtype=float)
        summary[name] = {
            "learned_seed_values": learned.tolist(),
            "learned_mean": float(learned.mean()),
            "learned_seed_bootstrap_95": seed_bootstrap_interval(learned, rng),
            "random_seed_values": random.tolist(),
            "random_mean": float(random.mean()),
            "learned_minus_random_seed_values": delta.tolist(),
            "learned_minus_random_mean": float(delta.mean()),
            "learned_minus_random_seed_bootstrap_95": seed_bootstrap_interval(
                delta, rng
            ),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--filename", default="task_representation_probe_aggregate.json"
    )
    args = parser.parse_args()
    if Path(args.filename).name != args.filename or not args.filename.endswith(".json"):
        raise ValueError("--filename must be a JSON basename")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    rng = np.random.default_rng(20260828)
    methods = []
    for experiment in config["experiments"]:
        records = []
        for seed in config["seeds"]:
            path = (
                root / experiment["method"] / f"seed_{seed}"
                / "task_representation_probe.json"
            )
            records.append(validate_record(
                json.loads(path.read_text(encoding="utf-8")),
                experiment, int(seed), config,
            ))
        if len({
            json.dumps(record["probe_source_sha256"], sort_keys=True)
            for record in records
        }) != 1:
            raise ValueError("task probe seeds use inconsistent probe source")
        methods.append({
            "method": experiment["method"],
            "training_seeds": len(records),
            "metrics": summarize(records, rng),
            "seed_results": records,
        })
    payload = {
        "schema_version": 1,
        "protocol": AGGREGATE_PROTOCOL,
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "experiment": config["name"],
        "methods": methods,
    }
    atomic_json(payload, root / args.filename)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
