#!/usr/bin/env python3
"""Compare goal-resolution decodability on byte-identical held-out pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_representation_probes import seed_bootstrap_interval
from aggregate_visual_task_representation_probes import AGGREGATE_PROTOCOL, METRICS


def atomic_text(content, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _dataset_signature(record):
    dataset = record["probe_dataset"]
    return {
        key: dataset[key] for key in (
            "behavior_checkpoint", "behavior_method",
            "behavior_checkpoint_global_step", "behavior_observation_contract",
            "train_seed", "test_seed", "train_sha256", "test_sha256",
        )
    }


def _probe_signature(record):
    return {
        "probe_source_sha256": record["probe_source_sha256"],
        "train_samples": record["train_samples"],
        "test_samples": record["test_samples"],
        "ridge_regularization": record["ridge_regularization"],
        "targets": record["targets"],
    }


def compare(config):
    required = int(config["required_training_seeds"])
    semantics = config["benchmark_semantics"]
    methods = {}
    records_by_name = {}
    for entry in config["methods"]:
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        if payload.get("protocol") != AGGREGATE_PROTOCOL:
            raise ValueError("task-representation aggregate protocol mismatch")
        if payload.get("benchmark_semantics") != semantics:
            raise ValueError("task-representation aggregate semantics mismatch")
        matches = [
            item for item in payload.get("methods", [])
            if item.get("method") == entry["method"]
        ]
        if len(matches) != 1:
            raise ValueError("task-representation method is not unique")
        records = sorted(matches[0]["seed_results"], key=lambda item: item["training_seed"])
        if len(records) != required or len({r["training_seed"] for r in records}) != required:
            raise ValueError("task-representation comparison has wrong seed count")
        result = {
            "name": entry["name"], "method": entry["method"], "metrics": {},
        }
        for metric_name, (metric_key, delta_key) in METRICS.items():
            learned = np.asarray([
                record["learned_encoder"][metric_key] for record in records
            ], dtype=float)
            random = np.asarray([
                record["random_encoder"][metric_key] for record in records
            ], dtype=float)
            delta = np.asarray([record[delta_key] for record in records], dtype=float)
            if not np.isfinite(np.concatenate((learned, random, delta))).all():
                raise ValueError("task-representation comparison contains non-finite metrics")
            if not np.allclose(delta, learned - random, rtol=0.0, atol=1e-12):
                raise ValueError("task-representation metric arithmetic mismatch")
            rng = np.random.default_rng(20260828)
            result["metrics"][metric_name] = {
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
        methods[entry["name"]] = result
        records_by_name[entry["name"]] = records

    names = list(methods)
    reference = records_by_name[names[0]]
    seeds = [record["training_seed"] for record in reference]
    for name in names[1:]:
        candidates = records_by_name[name]
        if [record["training_seed"] for record in candidates] != seeds:
            raise ValueError("task-representation methods use different seeds")
        for seed, left, right in zip(seeds, reference, candidates):
            if _dataset_signature(left) != _dataset_signature(right):
                raise ValueError(f"task-representation pixels differ for seed {seed}")
            if _probe_signature(left) != _probe_signature(right):
                raise ValueError(f"task-representation protocols differ for seed {seed}")
            for metric_key, _ in METRICS.values():
                if not np.isclose(
                    left["random_encoder"][metric_key],
                    right["random_encoder"][metric_key], rtol=0.0, atol=1e-12,
                ):
                    raise ValueError("matched task-probe random controls disagree")

    primary = config["primary_diagnostic"]
    treatment = methods[primary["treatment"]]
    control = methods[primary["control"]]
    paired = {}
    for metric_name in METRICS:
        treatment_values = np.asarray(
            treatment["metrics"][metric_name]["learned_seed_values"], dtype=float
        )
        control_values = np.asarray(
            control["metrics"][metric_name]["learned_seed_values"], dtype=float
        )
        differences = treatment_values - control_values
        paired[metric_name] = {
            "paired_seed_differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "paired_seed_bootstrap_95": seed_bootstrap_interval(
                differences, np.random.default_rng(20260828)
            ),
        }
    return {
        "schema_version": 1,
        "protocol": "cross-method matched-pixel task-representation comparison",
        "benchmark_semantics": semantics,
        "training_seeds": seeds,
        "dataset_match_verified": True,
        "methods": list(methods.values()),
        "primary_diagnostic": {
            "treatment": primary["treatment"], "control": primary["control"],
            "paired_metrics": paired,
            "claim_boundary": (
                "goal-resolution linear decodability is task-semantic diagnostic "
                "evidence; supervised progress labels train both encoders, and a "
                "probe difference does not establish causal control benefit"
            ),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def markdown(payload):
    lines = [
        "# Matched-pixel task-semantic representation diagnostics", "",
        "The probe predicts two goal-resolution bits from byte-identical RGB datasets.", "",
        "| Encoder | Balanced accuracy | ROC AUC | R² |", "|---|---:|---:|---:|",
    ]
    for method in payload["methods"]:
        metrics = method["metrics"]
        lines.append(
            f"| {method['name']} | {metrics['balanced_accuracy']['learned_mean']:.4f} | "
            f"{metrics['roc_auc']['learned_mean']:.4f} | "
            f"{metrics['r2']['learned_mean']:.4f} |"
        )
    primary = payload["primary_diagnostic"]
    lines.extend([
        "",
        f"## Paired {primary['treatment']} − {primary['control']} diagnostic",
        "",
    ])
    for name, result in payload["primary_diagnostic"]["paired_metrics"].items():
        interval = result["paired_seed_bootstrap_95"]
        lines.append(
            f"- {name}: {result['mean_difference']:+.4f} "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}]"
        )
    lines.extend(["", f"Claim boundary: {primary['claim_boundary']}.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare(json.loads(Path(args.config).read_text(encoding="utf-8")))
    root = Path(args.output)
    atomic_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        root / "task_representation_comparison.json",
    )
    atomic_text(markdown(payload), root / "task_representation_comparison.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
