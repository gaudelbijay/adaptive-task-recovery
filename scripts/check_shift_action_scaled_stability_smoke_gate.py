#!/usr/bin/env python3
"""Fail closed on the mechanically scaled V25 shift-consistency smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from check_drac_stability_smoke_gate import evaluations, load, normalized, score
from check_shift_action_stability_smoke_gate import bounded_loss_check


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {
    "trainer": Path(__file__).with_name(
        "train_visual_recovery_dual_teacher_shift_action_ppo.py"
    ),
    "trainer_wrapper": Path(__file__).with_name(
        "train_visual_recovery_dual_teacher_shift_action_ppo.py"
    ),
    "base_trainer": Path(__file__).with_name(
        "train_visual_recovery_dual_teacher_drac_ppo.py"
    ),
    "environment": ROOT / "src/atr/envs/learned_recovery.py",
    "environment_v3": ROOT / "src/atr/envs/learned_recovery_v3.py",
    "bounded_shift_action_consistency": Path(__file__).with_name(
        "bounded_shift_action_consistency.py"
    ),
}


def validate_loaded_configs(
    baseline: dict,
    candidate: dict,
    extension: dict,
    *,
    seed: int,
    expected_coefficient: float,
) -> tuple[dict, dict, dict]:
    if normalized(baseline, remove_budget=True) != normalized(
        candidate, remove_budget=True
    ):
        raise ValueError("V25 smoke differs from V19 outside treatment/budget fields")
    if normalized(baseline, remove_budget=False) != normalized(
        extension, remove_budget=False
    ):
        raise ValueError("V25 extension differs from V19 outside treatment fields")
    for value in (baseline, candidate, extension):
        if len(value.get("experiments", [])) != 1:
            raise ValueError("scaled gate requires exactly one experiment per config")
    if candidate.get("seeds") != [seed] or seed not in baseline.get("seeds", []):
        raise ValueError("V25 smoke must retain the paired V19 seed")
    if extension.get("seeds") != baseline.get("seeds"):
        raise ValueError("V25 extension must retain every V19 seed")
    baseline_task = baseline["experiments"][0]
    candidate_task = candidate["experiments"][0]
    extension_task = extension["experiments"][0]
    if int(baseline_task.get("augmentation_pad", -1)) != 0:
        raise ValueError("unexpected V19 augmentation setting")
    if "drac_policy_coefficient" in baseline_task:
        raise ValueError("V19 unexpectedly contains policy consistency")
    for task in (candidate_task, extension_task):
        if int(task.get("augmentation_pad", -1)) != 4:
            raise ValueError("V25 requires random-shift pad 4")
        if float(task.get("drac_policy_coefficient", -1)) != expected_coefficient:
            raise ValueError("V25 has the wrong mechanically scaled coefficient")
    return baseline_task, candidate_task, extension_task


def check(config: dict) -> dict:
    baseline, baseline_hash = load(config["baseline_config"])
    candidate, candidate_hash = load(config["candidate_config"])
    extension, extension_hash = load(config["extension_config"])
    seed = int(config["seed"])
    expected_coefficient = float(config["expected_drac_policy_coefficient"])
    if expected_coefficient != 0.02:
        raise ValueError("frozen V25 coefficient must equal 0.02")
    baseline_task, candidate_task, _ = validate_loaded_configs(
        baseline,
        candidate,
        extension,
        seed=seed,
        expected_coefficient=expected_coefficient,
    )

    batch = int(candidate_task["num_envs"]) * int(candidate_task["num_steps"])
    scheduled = int(candidate_task["total_timesteps"]) // batch * batch
    if scheduled != 19_996_672:
        raise ValueError("V25 smoke has the wrong floor-aligned budget")
    root = Path(config["results_root"])
    baseline_dir = root / baseline["name"] / baseline_task["method"] / f"seed_{seed}"
    candidate_dir = root / candidate["name"] / candidate_task["method"] / f"seed_{seed}"
    completion, completion_hash = load(candidate_dir / "TRAINING_COMPLETE.json")
    if int(completion.get("global_step", -1)) != scheduled:
        raise ValueError("V25 did not complete the exact smoke budget")

    baseline_records = evaluations(baseline_dir / "metrics.jsonl", scheduled)
    candidate_records = evaluations(candidate_dir / "metrics.jsonl", scheduled)
    tail_count = int(config["tail_evaluations"])
    if tail_count < 2 or min(len(baseline_records), len(candidate_records)) < tail_count:
        raise ValueError("V25 has insufficient frozen-tail evaluations")
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

    checkpoint_path = candidate_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_source = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in SOURCE_PATHS.items()
    }
    if checkpoint.get("source_sha256") != expected_source:
        raise ValueError("V25 checkpoint source does not match its frozen implementation")
    loss_check = bounded_loss_check(candidate_dir / "metrics.jsonl", scheduled)

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
        "finite_bounded_consistency": loss_check["finite_bounded_consistency"],
    }
    return {
        "schema_version": 1,
        "protocol": (
            "pre-held-out matched-budget scaled bounded shift-action stability "
            "allocation gate"
        ),
        "seed": seed,
        "expected_drac_policy_coefficient": expected_coefficient,
        "scheduled_step": scheduled,
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
        "maximum_logged_consistency_loss": loss_check[
            "maximum_logged_consistency_loss"
        ],
        "consistency_records": loss_check["consistency_records"],
        "checks": checks,
        "eligible": all(checks.values()),
        "thresholds": thresholds,
        "candidate_training_source_sha256": expected_source,
        "candidate_best_checkpoint_sha256": hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest(),
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
