#!/usr/bin/env python3
"""Evaluate V35 learned translation repair with explicit accounting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v35_translation_repair_agent as repaired


def install_protocol_adapter() -> None:
    original_atomic_json = base.atomic_json
    base.VisualAgent = repaired.TranslationRepairedV34Agent

    def annotated_atomic_json(payload: dict, path: Path) -> None:
        completion = json.loads((path.parent / "TRAINING_COMPLETE.json").read_text())
        if completion.get("training_protocol") != "supervised_translation_repair_v34":
            raise ValueError("V35 evaluation requires translation-repair provenance")
        local = int(completion["translation_training_transitions"])
        if int(payload["checkpoint_global_step"]) != local:
            raise ValueError("V35 checkpoint/transition accounting mismatch")
        agent = repaired.LAST_TRANSLATION_AGENT
        if agent is None:
            raise RuntimeError("V35 evaluator did not instantiate the repaired agent")
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
        payload["protocol_environment_transitions_consumed"] = int(
            completion["total_simulator_transitions"]
        )
        payload["training_protocol"] = completion["training_protocol"]
        payload["translation_training_transitions"] = local
        payload["initialization_simulator_transitions"] = int(
            completion["initialization_simulator_transitions"]
        )
        payload["ppo_accounting_applicable"] = False
        payload["learned_translation_route_fraction"] = agent.learned_translation_route_fraction
        payload["mean_routed_translation_magnitude"] = agent.mean_routed_translation_magnitude
        payload["v34_canonical_route_fraction"] = agent.robust.learned_route_fraction
        payload["domain_label_available_to_actor"] = False
        payload["evaluation_source_sha256"]["v35_protocol_adapter"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        original_atomic_json(payload, path)

    base.atomic_json = annotated_atomic_json


if __name__ == "__main__":
    install_protocol_adapter()
    base.main()
