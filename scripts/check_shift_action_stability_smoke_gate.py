#!/usr/bin/env python3
"""Fail-closed 20M allocation gate for bounded shift-action consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from check_drac_stability_smoke_gate import check as base_check


ROOT = Path(__file__).resolve().parents[1]
SMOKE_SOURCE_PATHS = {
    "trainer_wrapper": Path(__file__).with_name("archive")
    / "train_visual_recovery_dual_teacher_shift_action_ppo_smoke.py",
    "base_trainer": Path(__file__).with_name(
        "train_visual_recovery_dual_teacher_drac_ppo.py"
    ),
    "environment": ROOT / "src/atr/envs/learned_recovery.py",
    "environment_v3": ROOT / "src/atr/envs/learned_recovery_v3.py",
    "bounded_shift_action_consistency": Path(__file__).with_name(
        "bounded_shift_action_consistency.py"
    ),
}
EXTENSION_SOURCE_PATHS = {
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


def bounded_loss_check(path: str | Path, maximum_step: int) -> dict:
    values = []
    for line in Path(path).read_text().splitlines():
        record = json.loads(line)
        if "train_loss" in record and int(record["global_step"]) <= maximum_step:
            values.append(float(record["train_loss"]["drac_policy"]))
    if not values:
        raise ValueError("V24 smoke has no eligible consistency-loss records")
    maximum = max(values)
    return {
        "finite_bounded_consistency": all(
            math.isfinite(value) and 0 <= value <= 1.95 + 1e-6
            for value in values
        ),
        "maximum_logged_consistency_loss": maximum,
        "consistency_records": len(values),
    }


def check(config: dict) -> dict:
    payload = base_check(config)
    candidate = json.loads(Path(config["candidate_config"]).read_text())
    task = candidate["experiments"][0]
    seed = int(config["seed"])
    checkpoint_path = (
        Path(config["results_root"]) / candidate["name"] / task["method"]
        / f"seed_{seed}" / "best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    recorded_source = checkpoint.get("source_sha256")
    expected_smoke_source = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in SMOKE_SOURCE_PATHS.items()
    }
    if recorded_source != expected_smoke_source:
        raise ValueError("V24 smoke checkpoint source does not match frozen archive")
    loss_check = bounded_loss_check(
        checkpoint_path.with_name("metrics.jsonl"), int(payload["scheduled_step"]),
    )
    payload["checks"]["finite_bounded_consistency"] = loss_check[
        "finite_bounded_consistency"
    ]
    payload["eligible"] = all(payload["checks"].values())
    payload.update({
        key: value for key, value in loss_check.items()
        if key != "finite_bounded_consistency"
    })
    payload["protocol"] = (
        "pre-held-out matched-budget bounded shift-action stability allocation gate"
    )
    payload["candidate_training_source_sha256"] = recorded_source
    payload["candidate_best_checkpoint_sha256"] = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()
    payload["extension_launch_source_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in EXTENSION_SOURCE_PATHS.items()
    }
    return payload


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
