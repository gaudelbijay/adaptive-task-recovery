#!/usr/bin/env python3
"""Evaluate the RGB-routed V32 hybrid without modifying shared evaluators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v32_hybrid_domain_agent as hybrid


def install_protocol_adapter() -> None:
    original_atomic_json = base.atomic_json
    base.VisualAgent = hybrid.HybridDomainAgent

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion = json.loads(
            (path.parent / "TRAINING_COMPLETE.json").read_text(encoding="utf-8")
        )
        if completion.get("training_protocol") != "v19_preserving_geometry_routed_dagger":
            raise ValueError("V32 evaluation requires geometry-routed provenance")
        student = int(completion["student_transitions"])
        simulator = int(completion["simulator_transitions"])
        if int(payload["checkpoint_global_step"]) != student:
            raise ValueError("V32 checkpoint/transition accounting mismatch")
        if hybrid.LAST_HYBRID_AGENT is None:
            raise RuntimeError("V32 evaluator did not instantiate the hybrid agent")
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
        payload["learned_rgb_adapter_route_fraction"] = (
            hybrid.LAST_HYBRID_AGENT.learned_route_fraction
        )
        payload["domain_label_available_to_actor"] = False
        payload["training_only_geometry_target"] = True
        payload["evaluation_source_sha256"]["v32_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
