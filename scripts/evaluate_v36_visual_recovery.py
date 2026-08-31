#!/usr/bin/env python3
"""Evaluate V36 with explicit non-PPO interaction accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v36_continuous_canonical_agent as v36


def install_protocol_adapter() -> None:
    original_atomic_json = base.atomic_json
    base.VisualAgent = v36.ContinuousCanonicalV19Agent

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion = json.loads((path.parent / "TRAINING_COMPLETE.json").read_text())
        if completion.get("training_protocol") != "continuous_similarity_photometric_repair_v19":
            raise ValueError("V36 evaluation requires continuous-canonical provenance")
        local = int(completion["canonical_training_transitions"])
        if int(payload["checkpoint_global_step"]) != local:
            raise ValueError("V36 checkpoint/transition accounting mismatch")
        agent = v36.LAST_V36_AGENT
        if agent is None:
            raise RuntimeError("V36 evaluator did not instantiate its agent")
        for key in (
            "online_ppo_environment_steps", "initialization_ppo_environment_steps",
            "ppo_environment_steps", "online_protocol_ppo_environment_steps",
            "initialization_protocol_ppo_environment_steps", "protocol_ppo_environment_steps",
            "local_bc_dagger_environment_transitions",
            "initialization_bc_dagger_environment_transitions",
            "bc_dagger_environment_transitions",
        ):
            payload[key] = 0
        payload["total_environment_transitions"] = int(completion["total_simulator_transitions"])
        payload["protocol_environment_transitions_consumed"] = int(completion["total_simulator_transitions"])
        payload["training_protocol"] = completion["training_protocol"]
        payload["canonical_training_transitions"] = local
        payload["initialization_simulator_transitions"] = int(completion["initialization_simulator_transitions"])
        payload["ppo_accounting_applicable"] = False
        payload["learned_continuous_route_fraction"] = agent.learned_route_fraction
        payload["domain_label_available_to_actor"] = False
        payload["evaluation_source_sha256"]["v36_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
