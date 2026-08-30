#!/usr/bin/env python3
"""Evaluate V33 learned RGB canonicalization with explicit accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v33_canonical_view_agent as canonical


def install_protocol_adapter() -> None:
    original_atomic_json = base.atomic_json
    base.VisualAgent = canonical.CanonicalizedV19Agent

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion = json.loads(
            (path.parent / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
        )
        if completion.get("training_protocol") != "paired_canonical_view_v19_control":
            raise ValueError("V33 evaluation requires canonical-view provenance")
        student = int(completion["student_transitions"])
        simulator = int(completion["simulator_transitions"])
        if int(payload["checkpoint_global_step"]) != student:
            raise ValueError("V33 checkpoint/transition accounting mismatch")
        if canonical.LAST_CANONICAL_AGENT is None:
            raise RuntimeError("V33 evaluator did not instantiate the canonical agent")
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
        payload["paired_view_training_samples"] = int(
            completion["paired_view_training_samples"]
        )
        payload["local_bc_dagger_environment_transitions"] = student
        payload["bc_dagger_environment_transitions"] = student
        payload["ppo_accounting_applicable"] = False
        payload["learned_rgb_canonical_route_fraction"] = (
            canonical.LAST_CANONICAL_AGENT.learned_route_fraction
        )
        payload["domain_label_available_to_actor"] = False
        payload["evaluation_source_sha256"]["v33_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
