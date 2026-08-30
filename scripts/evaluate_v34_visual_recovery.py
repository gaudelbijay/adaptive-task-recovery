#!/usr/bin/env python3
"""Evaluate V34 factorized RGB canonicalization with explicit accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v34_factorized_canonical_agent as factorized


def install_protocol_adapter() -> None:
    original_atomic_json = base.atomic_json
    base.VisualAgent = factorized.FactorizedCanonicalV19Agent

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion = json.loads(
            (path.parent / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
        )
        if completion.get("training_protocol") != "factorized_canonical_v19_control":
            raise ValueError("V34 evaluation requires factorized-canonical provenance")
        primary = int(completion["student_transitions"])
        simulator = int(completion["simulator_transitions"])
        if int(payload["checkpoint_global_step"]) != primary:
            raise ValueError("V34 checkpoint/transition accounting mismatch")
        if factorized.LAST_FACTORIZED_AGENT is None:
            raise RuntimeError("V34 evaluator did not instantiate the factorized agent")
        for key in (
            "online_ppo_environment_steps", "initialization_ppo_environment_steps",
            "ppo_environment_steps", "online_protocol_ppo_environment_steps",
            "initialization_protocol_ppo_environment_steps", "protocol_ppo_environment_steps",
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
        payload["local_bc_dagger_environment_transitions"] = primary
        payload["bc_dagger_environment_transitions"] = primary
        payload["ppo_accounting_applicable"] = False
        payload["learned_factorized_canonical_route_fraction"] = (
            factorized.LAST_FACTORIZED_AGENT.learned_route_fraction
        )
        payload["domain_label_available_to_actor"] = False
        payload["evaluation_source_sha256"]["v34_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
