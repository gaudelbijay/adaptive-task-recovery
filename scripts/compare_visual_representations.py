#!/usr/bin/env python3
"""Compare visual encoders on an exactly matched held-out RGB dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_representation_probes import seed_bootstrap_interval


PROTOCOL = "held-out linear pose probe aggregated over training seeds"


def atomic_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def select_method(payload, method, semantics, seeds):
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("representation aggregate protocol mismatch")
    if payload.get("benchmark_semantics") != semantics:
        raise ValueError("representation aggregate semantics mismatch")
    matches = [item for item in payload.get("methods", []) if item.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one representation method: {method}")
    result = matches[0]
    records = sorted(result.get("seed_results", []), key=lambda item: item["training_seed"])
    if len(records) != seeds or int(result.get("training_seeds", -1)) != seeds:
        raise ValueError("representation comparison has the wrong training-seed count")
    if len({record["training_seed"] for record in records}) != seeds:
        raise ValueError("representation comparison has duplicate training seeds")
    return result, records


def dataset_signature(record):
    dataset = record["probe_dataset"]
    return {
        key: dataset[key] for key in (
            "behavior_checkpoint", "behavior_method",
            "behavior_checkpoint_global_step", "behavior_observation_contract",
            "train_seed", "test_seed", "train_sha256", "test_sha256",
        )
    }


def probe_signature(record):
    return {
        "probe_source_sha256": record["probe_source_sha256"],
        "train_samples": record["train_samples"],
        "test_samples": record["test_samples"],
        "ridge_regularization": record["ridge_regularization"],
        "targets": record["targets"],
    }


def compare(config):
    semantics = config["benchmark_semantics"]
    required_seeds = int(config["required_training_seeds"])
    methods = {}
    records_by_name = {}
    for entry in config["methods"]:
        payload = json.loads(Path(entry["path"]).read_text(encoding="utf-8"))
        result, records = select_method(
            payload, entry["method"], semantics, required_seeds,
        )
        learned = np.asarray([
            record["learned_encoder"]["r2_variance_weighted"] for record in records
        ], dtype=float)
        random = np.asarray([
            record["random_encoder"]["r2_variance_weighted"] for record in records
        ], dtype=float)
        advantages = np.asarray([
            record["learned_minus_random_r2"] for record in records
        ], dtype=float)
        if not np.isfinite(np.concatenate((learned, random, advantages))).all():
            raise ValueError("representation comparison contains non-finite R2 values")
        if not np.allclose(advantages, learned - random, rtol=0.0, atol=1e-12):
            raise ValueError("representation comparison has inconsistent R2 arithmetic")
        rng = np.random.default_rng(20260828)
        methods[entry["name"]] = {
            "name": entry["name"],
            "method": entry["method"],
            "learned_r2_seed_values": learned.tolist(),
            "learned_r2_mean": float(learned.mean()),
            "learned_r2_seed_bootstrap_95": seed_bootstrap_interval(learned, rng),
            "random_r2_seed_values": random.tolist(),
            "random_r2_mean": float(random.mean()),
            "learned_minus_random_r2_seed_values": advantages.tolist(),
            "learned_minus_random_r2_mean": float(advantages.mean()),
            "learned_minus_random_r2_seed_bootstrap_95": seed_bootstrap_interval(
                advantages, rng,
            ),
        }
        records_by_name[entry["name"]] = records

    names = list(methods)
    reference_records = records_by_name[names[0]]
    reference_seeds = [record["training_seed"] for record in reference_records]
    for name in names[1:]:
        records = records_by_name[name]
        if [record["training_seed"] for record in records] != reference_seeds:
            raise ValueError("representation methods do not use identical training seeds")
        for seed, reference, candidate in zip(reference_seeds, reference_records, records):
            if dataset_signature(candidate) != dataset_signature(reference):
                raise ValueError(
                    f"representation methods do not use identical pixels for seed {seed}"
                )
            if probe_signature(candidate) != probe_signature(reference):
                raise ValueError(
                    f"representation methods use different probe protocols for seed {seed}"
                )
            random_left = float(reference["random_encoder"]["r2_variance_weighted"])
            random_right = float(candidate["random_encoder"]["r2_variance_weighted"])
            if not np.isclose(random_left, random_right, rtol=0.0, atol=1e-12):
                raise ValueError("matched random-encoder controls disagree")

    primary = config["primary_diagnostic"]
    treatment = methods[primary["treatment"]]
    control = methods[primary["control"]]
    differences = (
        np.asarray(treatment["learned_r2_seed_values"], dtype=float)
        - np.asarray(control["learned_r2_seed_values"], dtype=float)
    )
    interval = seed_bootstrap_interval(
        differences, np.random.default_rng(20260828),
    )
    diagnostic = {
        "treatment": primary["treatment"],
        "control": primary["control"],
        "paired_seed_r2_differences": differences.tolist(),
        "mean_r2_difference": float(differences.mean()),
        "paired_seed_bootstrap_95": interval,
        "positive_interval": interval[0] > 0,
        "claim_boundary": (
            "linear decodability on matched pixels is diagnostic representation "
            "evidence; it does not by itself establish a causal control benefit"
        ),
    }
    return {
        "schema_version": 1,
        "protocol": "cross-method matched-pixel visual representation comparison",
        "benchmark_semantics": semantics,
        "training_seeds": reference_seeds,
        "dataset_match_verified": True,
        "methods": list(methods.values()),
        "primary_diagnostic": diagnostic,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def markdown(payload):
    lines = [
        "# Matched-pixel visual representation diagnostics", "",
        "All encoders are probed on byte-identical RGB/label datasets within each seed.", "",
        "| Method | Learned R² | Random R² | Learned − random R² |", "|---|---:|---:|---:|",
    ]
    for item in payload["methods"]:
        lines.append(
            f"| {item['name']} | {item['learned_r2_mean']:.4f} | "
            f"{item['random_r2_mean']:.4f} | {item['learned_minus_random_r2_mean']:.4f} |"
        )
    result = payload["primary_diagnostic"]
    lines.extend([
        "", "## Temporal-SSL diagnostic", "",
        f"Paired mean R² difference: {result['mean_r2_difference']:+.4f}; "
        f"seed-bootstrap 95% interval [{result['paired_seed_bootstrap_95'][0]:+.4f}, "
        f"{result['paired_seed_bootstrap_95'][1]:+.4f}].",
        "", f"Claim boundary: {result['claim_boundary']}.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/visual_representation_comparison_v1.json",
    )
    parser.add_argument("--output", default="results/final_visual_comparison")
    args = parser.parse_args()
    payload = compare(json.loads(Path(args.config).read_text(encoding="utf-8")))
    root = Path(args.output)
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", root / "representation_comparison.json")
    atomic_text(markdown(payload), root / "representation_comparison.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
