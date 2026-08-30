#!/usr/bin/env python3
"""Fail-closed runtime gate for bounded shift-action consistency."""

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


def normalized(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.pop("name", None)
    value.pop("claim_boundary", None)
    value["experiments"][0].pop("method", None)
    return value


def check(config: dict) -> dict:
    runtime_ref, runtime_ref_hash = load(config["reference_runtime_config"])
    runtime, runtime_hash = load(config["candidate_runtime_config"])
    smoke_ref, smoke_ref_hash = load(config["reference_smoke_config"])
    smoke, smoke_hash = load(config["candidate_smoke_config"])
    if normalized(runtime_ref) != normalized(runtime):
        raise ValueError("V24 runtime differs from V22 runtime outside method identity")
    if normalized(smoke_ref) != normalized(smoke):
        raise ValueError("V24 smoke differs from V22 smoke outside method identity")
    seed = int(config["seed"])
    if runtime.get("seeds") != [seed] or smoke.get("seeds") != [seed]:
        raise ValueError("runtime and smoke must retain the frozen single seed")
    task = runtime["experiments"][0]
    batch = int(task["num_envs"]) * int(task["num_steps"])
    scheduled = int(task["total_timesteps"]) // batch * batch
    run_dir = (
        Path(config["results_root"]) / runtime["name"] / task["method"]
        / f"seed_{seed}"
    )
    completion, completion_hash = load(run_dir / "TRAINING_COMPLETE.json")
    if int(completion.get("global_step", -1)) != scheduled:
        raise ValueError("V24 runtime did not complete its exact budget")
    records = [
        json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    evaluations = [record for record in records if "eval" in record]
    training = [record for record in records if "train_loss" in record]
    if len(evaluations) < 2 or not training:
        raise ValueError("V24 runtime lacks final evaluation or training diagnostics")
    if int(evaluations[0]["global_step"]) != 0:
        raise ValueError("V24 runtime lacks the frozen initialization evaluation")
    if int(evaluations[-1]["global_step"]) > scheduled:
        raise ValueError("V24 final evaluation exceeds its runtime budget")
    losses = [float(record["train_loss"]["drac_policy"]) for record in training]
    thresholds = config["thresholds"]
    initial_success = float(evaluations[0]["eval"]["success_at_end"])
    final_success = float(evaluations[-1]["eval"]["success_at_end"])
    final_violation = float(evaluations[-1]["eval"]["constraint_violated"])
    maximum_loss = max(losses)
    checks = {
        "finite_consistency": all(math.isfinite(value) and value >= 0 for value in losses),
        "bounded_consistency": maximum_loss
        <= float(thresholds["maximum_logged_consistency_loss"]) + 1e-6,
        "final_success": final_success
        >= float(thresholds["minimum_final_success_at_end"]),
        "final_violation": final_violation
        <= float(thresholds["maximum_final_constraint_violation"]),
        "success_retention": initial_success - final_success
        <= float(thresholds["maximum_success_drop_from_initial"]),
    }
    return {
        "schema_version": 1,
        "protocol": "bounded shift-action runtime allocation gate",
        "seed": seed,
        "scheduled_step": scheduled,
        "initial_success_at_end": initial_success,
        "final_success_at_end": final_success,
        "final_constraint_violation": final_violation,
        "maximum_logged_consistency_loss": maximum_loss,
        "checks": checks,
        "eligible": all(checks.values()),
        "thresholds": thresholds,
        "source_sha256": {
            config["reference_runtime_config"]: runtime_ref_hash,
            config["candidate_runtime_config"]: runtime_hash,
            config["reference_smoke_config"]: smoke_ref_hash,
            config["candidate_smoke_config"]: smoke_hash,
            str(run_dir / "TRAINING_COMPLETE.json"): completion_hash,
        },
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.config)
    raw = path.read_bytes()
    payload = check(json.loads(raw))
    payload["config"] = str(path)
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
