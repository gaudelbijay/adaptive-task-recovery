#!/usr/bin/env python3
"""Fail-closed allocation gate for the V21 low-variance development smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path


def load(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def normalized_smoke(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("name", None)
    value.pop("seeds", None)
    value.pop("claim_boundary", None)
    experiment = value["experiments"][0]
    for key in ("method", "total_timesteps", "temporal_variance_coefficient"):
        experiment.pop(key, None)
    return value


def normalized_extension(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("name", None)
    value.pop("claim_boundary", None)
    experiment = value["experiments"][0]
    for key in ("method", "temporal_variance_coefficient"):
        experiment.pop(key, None)
    return value


def evaluation_records(path: Path, maximum_step: int) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "eval" in record and int(record["global_step"]) <= maximum_step:
            records.append(record)
    if not records:
        raise ValueError(f"no eligible evaluation records: {path}")
    return records


def score(record: dict, penalty: float) -> float:
    metrics = record["eval"]
    success = float(metrics["success_at_end"])
    violation = float(metrics["constraint_violated"])
    result = success - penalty * violation + 1e-6 * float(metrics.get("return", 0.0))
    if not math.isfinite(result):
        raise ValueError("non-finite checkpoint-selection score")
    return result


def check(config: dict) -> dict:
    baseline, baseline_hash = load(config["baseline_config"])
    candidate, candidate_hash = load(config["candidate_config"])
    extension, extension_hash = load(config["extension_config"])
    if normalized_smoke(baseline) != normalized_smoke(candidate):
        raise ValueError("candidate differs from baseline outside frozen smoke fields")
    if normalized_extension(baseline) != normalized_extension(extension):
        raise ValueError("extension differs from baseline outside frozen extension fields")
    if len(baseline["experiments"]) != 1 or len(candidate["experiments"]) != 1:
        raise ValueError("gate requires exactly one experiment per config")
    seed = int(config["seed"])
    if seed not in baseline["seeds"] or candidate["seeds"] != [seed]:
        raise ValueError("candidate must contain only the paired baseline seed")
    baseline_task = baseline["experiments"][0]
    candidate_task = candidate["experiments"][0]
    extension_task = extension["experiments"][0]
    if float(baseline_task["temporal_variance_coefficient"]) != 0.01:
        raise ValueError("unexpected baseline variance coefficient")
    if float(candidate_task["temporal_variance_coefficient"]) != 0.001:
        raise ValueError("unexpected candidate variance coefficient")
    if float(extension_task["temporal_variance_coefficient"]) != 0.001:
        raise ValueError("unexpected extension variance coefficient")
    batch_size = int(candidate_task["num_envs"]) * int(candidate_task["num_steps"])
    scheduled_step = int(candidate_task["total_timesteps"]) // batch_size * batch_size
    root = Path(config["results_root"])

    def run_dir(experiment: dict, task: dict) -> Path:
        return root / experiment["name"] / task["method"] / f"seed_{seed}"

    baseline_dir = run_dir(baseline, baseline_task)
    candidate_dir = run_dir(candidate, candidate_task)
    completion, completion_hash = load(candidate_dir / "TRAINING_COMPLETE.json")
    if int(completion.get("global_step", -1)) != scheduled_step:
        raise ValueError("candidate did not complete the exact diagnostic budget")
    penalty = float(candidate["selection_failure_penalty"])
    baseline_best = max(
        evaluation_records(baseline_dir / "metrics.jsonl", scheduled_step),
        key=lambda item: score(item, penalty),
    )
    candidate_best = max(
        evaluation_records(candidate_dir / "metrics.jsonl", scheduled_step),
        key=lambda item: score(item, penalty),
    )
    baseline_score = score(baseline_best, penalty)
    candidate_score = score(candidate_best, penalty)
    metrics = candidate_best["eval"]
    thresholds = config["thresholds"]
    checks = {
        "success_at_end": float(metrics["success_at_end"]) >= float(
            thresholds["minimum_success_at_end"]
        ),
        "constraint_violation": float(metrics["constraint_violated"]) <= float(
            thresholds["maximum_constraint_violation"]
        ),
        "safety_weighted_improvement": candidate_score - baseline_score >= float(
            thresholds["minimum_safety_weighted_improvement"]
        ),
    }
    return {
        "schema_version": 1,
        "protocol": "pre-held-out one-seed VICReg stabilization allocation gate",
        "seed": seed,
        "scheduled_step": scheduled_step,
        "baseline_best_step": int(baseline_best["global_step"]),
        "candidate_best_step": int(candidate_best["global_step"]),
        "baseline_safety_weighted_score": baseline_score,
        "candidate_safety_weighted_score": candidate_score,
        "safety_weighted_improvement": candidate_score - baseline_score,
        "candidate_success_at_end": float(metrics["success_at_end"]),
        "candidate_constraint_violation": float(metrics["constraint_violated"]),
        "checks": checks,
        "eligible": all(checks.values()),
        "thresholds": thresholds,
        "source_sha256": {
            config["baseline_config"]: baseline_hash,
            config["candidate_config"]: candidate_hash,
            config["extension_config"]: extension_hash,
            str(candidate_dir / "TRAINING_COMPLETE.json"): completion_hash,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    payload = check(json.loads(config_bytes))
    payload["config"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
