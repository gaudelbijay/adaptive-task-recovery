#!/usr/bin/env python3
"""Fail-closed allocation gate for the matched V22 DrAC stability smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path


TREATMENT_FIELDS = ("method", "augmentation_pad", "drac_policy_coefficient")


def load(path: str | Path) -> tuple[dict, str]:
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def normalized(config: dict, *, remove_budget: bool) -> dict:
    value = copy.deepcopy(config)
    value.pop("name", None)
    value.pop("claim_boundary", None)
    if remove_budget:
        value.pop("seeds", None)
    experiment = value["experiments"][0]
    for key in TREATMENT_FIELDS:
        experiment.pop(key, None)
    if remove_budget:
        experiment.pop("total_timesteps", None)
    return value


def evaluations(path: Path, maximum_step: int) -> list[dict]:
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
    value = (
        float(metrics["success_at_end"])
        - penalty * float(metrics["constraint_violated"])
        + 1e-6 * float(metrics.get("return", 0.0))
    )
    if not math.isfinite(value):
        raise ValueError("non-finite stability score")
    return value


def check(config: dict) -> dict:
    baseline, baseline_hash = load(config["baseline_config"])
    candidate, candidate_hash = load(config["candidate_config"])
    extension, extension_hash = load(config["extension_config"])
    if normalized(baseline, remove_budget=True) != normalized(
        candidate, remove_budget=True,
    ):
        raise ValueError("candidate differs from V19 outside frozen treatment/budget fields")
    if normalized(baseline, remove_budget=False) != normalized(
        extension, remove_budget=False,
    ):
        raise ValueError("extension differs from V19 outside frozen treatment fields")
    for value in (baseline, candidate, extension):
        if len(value.get("experiments", [])) != 1:
            raise ValueError("gate requires exactly one experiment per config")
    seed = int(config["seed"])
    if candidate.get("seeds") != [seed] or seed not in baseline.get("seeds", []):
        raise ValueError("candidate must contain only the paired V19 seed")
    if extension.get("seeds") != baseline.get("seeds"):
        raise ValueError("extension and V19 seed sets differ")
    baseline_task = baseline["experiments"][0]
    candidate_task = candidate["experiments"][0]
    extension_task = extension["experiments"][0]
    if int(baseline_task.get("augmentation_pad", -1)) != 0:
        raise ValueError("unexpected V19 augmentation setting")
    if "drac_policy_coefficient" in baseline_task:
        raise ValueError("V19 unexpectedly contains DrAC policy consistency")
    for task in (candidate_task, extension_task):
        if int(task.get("augmentation_pad", -1)) != 4:
            raise ValueError("DrAC treatment requires random-shift pad 4")
        if float(task.get("drac_policy_coefficient", -1)) != 0.1:
            raise ValueError("DrAC treatment requires policy coefficient 0.1")

    batch = int(candidate_task["num_envs"]) * int(candidate_task["num_steps"])
    scheduled = int(candidate_task["total_timesteps"]) // batch * batch
    root = Path(config["results_root"])
    baseline_dir = (
        root / baseline["name"] / baseline_task["method"] / f"seed_{seed}"
    )
    candidate_dir = (
        root / candidate["name"] / candidate_task["method"] / f"seed_{seed}"
    )
    completion, completion_hash = load(candidate_dir / "TRAINING_COMPLETE.json")
    if int(completion.get("global_step", -1)) != scheduled:
        raise ValueError("candidate did not complete the exact smoke budget")
    baseline_records = evaluations(baseline_dir / "metrics.jsonl", scheduled)
    candidate_records = evaluations(candidate_dir / "metrics.jsonl", scheduled)
    tail_count = int(config["tail_evaluations"])
    if tail_count < 2 or len(baseline_records) < tail_count or len(candidate_records) < tail_count:
        raise ValueError("insufficient evaluation records for the frozen tail window")
    penalty = float(candidate["selection_failure_penalty"])
    baseline_best = max(baseline_records, key=lambda item: score(item, penalty))
    candidate_best = max(candidate_records, key=lambda item: score(item, penalty))
    baseline_best_score = score(baseline_best, penalty)
    candidate_best_score = score(candidate_best, penalty)
    baseline_tail = baseline_records[-tail_count:]
    candidate_tail = candidate_records[-tail_count:]
    baseline_tail_score = sum(score(item, penalty) for item in baseline_tail) / tail_count
    candidate_tail_score = sum(score(item, penalty) for item in candidate_tail) / tail_count
    candidate_tail_violation = sum(
        float(item["eval"]["constraint_violated"]) for item in candidate_tail
    ) / tail_count
    thresholds = config["thresholds"]
    checks = {
        "best_success_at_end": float(candidate_best["eval"]["success_at_end"])
        >= float(thresholds["minimum_best_success_at_end"]),
        "best_constraint_violation": float(
            candidate_best["eval"]["constraint_violated"]
        ) <= float(thresholds["maximum_best_constraint_violation"]),
        "best_score_margin": candidate_best_score - baseline_best_score
        >= float(thresholds["minimum_best_score_margin"]),
        "tail_mean_constraint_violation": candidate_tail_violation
        <= float(thresholds["maximum_tail_mean_constraint_violation"]),
        "tail_mean_score_improvement": candidate_tail_score - baseline_tail_score
        >= float(thresholds["minimum_tail_mean_score_improvement"]),
    }
    return {
        "schema_version": 1,
        "protocol": "pre-held-out matched-budget DrAC stability allocation gate",
        "seed": seed, "scheduled_step": scheduled,
        "baseline_best_step": int(baseline_best["global_step"]),
        "candidate_best_step": int(candidate_best["global_step"]),
        "baseline_best_score": baseline_best_score,
        "candidate_best_score": candidate_best_score,
        "best_score_margin": candidate_best_score - baseline_best_score,
        "baseline_tail_mean_score": baseline_tail_score,
        "candidate_tail_mean_score": candidate_tail_score,
        "tail_mean_score_improvement": candidate_tail_score - baseline_tail_score,
        "candidate_best_success_at_end": float(
            candidate_best["eval"]["success_at_end"]
        ),
        "candidate_best_constraint_violation": float(
            candidate_best["eval"]["constraint_violated"]
        ),
        "candidate_tail_mean_constraint_violation": candidate_tail_violation,
        "tail_evaluations": tail_count,
        "checks": checks, "eligible": all(checks.values()),
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
    raw = config_path.read_bytes()
    payload = check(json.loads(raw))
    payload["config"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(raw).hexdigest()
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
