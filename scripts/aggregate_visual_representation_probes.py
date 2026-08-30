#!/usr/bin/env python3
"""Aggregate held-out representation probes across independently trained seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


PROTOCOL = "held-out linear pose probe; labels unavailable to actor"
POSE_KEYS = (
    "critic_red_cube_pose", "critic_blue_cube_pose",
    "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
)
TARGETS = [f"{key}:{axis}" for key in POSE_KEYS for axis in ("x", "y", "z")]


def seed_bootstrap_interval(values, rng, repetitions=20000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("seed bootstrap requires at least one value")
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    samples = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


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


def validate_probe_record(
    record, experiment, seed, expected_samples=8192,
    behavior_checkpoint=None, expected_ridge=1.0,
):
    expected_semantics = (
        "event_reward_intervention_target_only_v3"
        if experiment["env_id"] == "LearnedRecovery-v3"
        else "intervention_target_only_v2"
    )
    expected = {
        "protocol": PROTOCOL,
        "benchmark_semantics": expected_semantics,
        "observation_contract": expected_observation_contract(experiment),
        "method": experiment["method"],
        "training_seed": seed,
        "train_samples": expected_samples,
        "test_samples": expected_samples,
        "ridge_regularization": float(expected_ridge),
        "targets": TARGETS,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"probe {key} mismatch: expected {value!r}, observed {record.get(key)!r}"
            )
    if int(record.get("checkpoint_global_step", -1)) < 0:
        raise ValueError("probe lacks checkpoint provenance")
    training_source = record.get("training_source_sha256")
    if not isinstance(training_source, dict) or not training_source.get("trainer"):
        raise ValueError("probe lacks training-source provenance")
    probe_source = record.get("probe_source_sha256")
    if not isinstance(probe_source, dict) or not probe_source.get("probe"):
        raise ValueError("probe lacks probe-source provenance")
    dataset = record.get("probe_dataset")
    if not isinstance(dataset, dict):
        raise ValueError("probe lacks matched-dataset provenance")
    expected_dataset = {
        "protocol": "frozen seed-matched behavior policy; identical pixels across methods",
        "train_seed": 93000000 + seed * 10,
        "test_seed": 93000000 + seed * 10 + 1,
    }
    for key, value in expected_dataset.items():
        if dataset.get(key) != value:
            raise ValueError(f"probe dataset {key} mismatch")
    if not behavior_checkpoint:
        raise ValueError("probe configuration lacks frozen behavior checkpoint")
    if dataset.get("behavior_checkpoint") != str(
        behavior_checkpoint
    ).format(seed=seed):
        raise ValueError("probe behavior checkpoint disagrees with configuration")
    for key in ("behavior_checkpoint", "behavior_method", "behavior_observation_contract",
                "train_sha256", "test_sha256"):
        if not dataset.get(key):
            raise ValueError(f"probe dataset lacks {key}")
    for key in ("train_sha256", "test_sha256"):
        digest = dataset[key]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"probe dataset {key} is not a SHA-256 digest")
    if int(dataset.get("behavior_checkpoint_global_step", -1)) < 0:
        raise ValueError("probe dataset lacks behavior-checkpoint step")
    behavior_source = dataset.get("behavior_training_source_sha256")
    if not isinstance(behavior_source, dict) or not behavior_source.get("trainer"):
        raise ValueError("probe dataset lacks behavior training-source provenance")
    learned = float(record["learned_encoder"]["r2_variance_weighted"])
    random = float(record["random_encoder"]["r2_variance_weighted"])
    difference = float(record["learned_minus_random_r2"])
    if not np.isfinite([learned, random, difference]).all():
        raise ValueError("probe contains non-finite representation metrics")
    if not np.isclose(difference, learned - random):
        raise ValueError("probe learned-minus-random metric is inconsistent")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--filename", default="representation_probe_aggregate.json")
    parser.add_argument("--probe-filename", default="representation_probe.json")
    args = parser.parse_args()
    if Path(args.filename).name != args.filename or not args.filename.endswith(".json"):
        raise ValueError("--filename must be a JSON basename")
    if Path(args.probe_filename).name != args.probe_filename or not args.probe_filename.endswith(".json"):
        raise ValueError("--probe-filename must be a JSON basename")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    rng = np.random.default_rng(20260828)
    methods = []
    semantics = set()
    for experiment in config["experiments"]:
        records = []
        for seed in config["seeds"]:
            path = root / experiment["method"] / f"seed_{seed}" / args.probe_filename
            if not path.exists():
                raise FileNotFoundError(path)
            record = validate_probe_record(
                json.loads(path.read_text(encoding="utf-8")),
                experiment, seed,
                int(config.get("representation_probe_samples_per_split", 8192)),
                config.get("representation_probe_behavior_checkpoint"),
                float(config.get("representation_probe_ridge", 1.0)),
            )
            records.append(record)
            semantics.add(record["benchmark_semantics"])
        if len({
            json.dumps(record["probe_source_sha256"], sort_keys=True)
            for record in records
        }) != 1:
            raise ValueError("probe seeds use inconsistent probe-source provenance")
        learned = [item["learned_encoder"]["r2_variance_weighted"] for item in records]
        random = [item["random_encoder"]["r2_variance_weighted"] for item in records]
        differences = [item["learned_minus_random_r2"] for item in records]
        methods.append({
            "method": experiment["method"],
            "training_seeds": len(records),
            "train_samples_per_seed": records[0]["train_samples"],
            "test_samples_per_seed": records[0]["test_samples"],
            "learned_r2_seed_values": learned,
            "learned_r2_mean": float(np.mean(learned)),
            "learned_r2_seed_bootstrap_95": seed_bootstrap_interval(learned, rng),
            "random_r2_seed_values": random,
            "random_r2_mean": float(np.mean(random)),
            "learned_minus_random_r2_seed_values": differences,
            "learned_minus_random_r2_mean": float(np.mean(differences)),
            "learned_minus_random_r2_seed_bootstrap_95": seed_bootstrap_interval(
                differences, rng
            ),
            "seed_results": records,
        })
    if len(semantics) != 1:
        raise ValueError("probe records mix benchmark semantics")
    payload = {
        "schema_version": 1,
        "protocol": "held-out linear pose probe aggregated over training seeds",
        "benchmark_semantics": semantics.pop(),
        "experiment": config["name"],
        "methods": methods,
    }
    path = root / args.filename
    atomic_json(payload, path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
