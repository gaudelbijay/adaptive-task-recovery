#!/usr/bin/env python3
"""Annotate V31 evaluation without changing frozen shared evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base


def install_protocol_adapter() -> None:
    """Replace legacy PPO accounting with V31's actual DAgger accounting."""

    original_atomic_json = base.atomic_json

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion_path = path.parent / "TRAINING_COMPLETE.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("training_protocol") != "full_episode_same_physics_multicamera_dagger":
            raise ValueError("V31 evaluation requires multicamera DAgger provenance")
        student = int(completion["student_transitions"])
        simulator = int(completion["simulator_transitions"])
        if int(payload["checkpoint_global_step"]) != student:
            raise ValueError("V31 checkpoint/DAgger-transition accounting mismatch")
        for key in (
            "online_ppo_environment_steps",
            "initialization_ppo_environment_steps",
            "ppo_environment_steps",
            "online_protocol_ppo_environment_steps",
            "initialization_protocol_ppo_environment_steps",
            "protocol_ppo_environment_steps",
            "local_bc_dagger_environment_transitions",
            "initialization_bc_dagger_environment_transitions",
            "bc_dagger_environment_transitions",
        ):
            payload[key] = 0
        payload["total_environment_transitions"] = simulator
        payload["protocol_environment_transitions_consumed"] = simulator
        payload["training_protocol"] = completion["training_protocol"]
        payload["dagger_environment_transitions"] = int(
            completion["dagger_environment_transitions"]
        )
        payload["local_bc_dagger_environment_transitions"] = student
        payload["bc_dagger_environment_transitions"] = student
        payload["ppo_accounting_applicable"] = False
        payload["evaluation_source_sha256"]["v31_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
