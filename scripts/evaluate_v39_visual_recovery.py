#!/usr/bin/env python3
"""Evaluate magnitude-gated V39 with explicit non-PPO accounting."""

import hashlib
import json
from pathlib import Path

import evaluate_visual_recovery_ppo as base
import v39_magnitude_gated_agent as v39


def install_protocol_adapter():
    original_atomic_json = base.atomic_json
    base.VisualAgent = v39.MagnitudeGatedDenseV19Agent

    def annotated(payload, path: Path):
        completion = json.loads((path.parent / "TRAINING_COMPLETE.json").read_text())
        if completion.get("training_protocol") != "backkey_targeted_dense_repair_v19":
            raise ValueError("V39 evaluation requires targeted-dense provenance")
        local = int(completion["dense_finetune_transitions"])
        if int(payload["checkpoint_global_step"]) != local: raise ValueError("V39 accounting mismatch")
        for key in ("online_ppo_environment_steps","initialization_ppo_environment_steps","ppo_environment_steps",
                    "online_protocol_ppo_environment_steps","initialization_protocol_ppo_environment_steps",
                    "protocol_ppo_environment_steps","local_bc_dagger_environment_transitions",
                    "initialization_bc_dagger_environment_transitions","bc_dagger_environment_transitions"):
            payload[key] = 0
        payload["total_environment_transitions"] = int(completion["total_simulator_transitions"])
        payload["protocol_environment_transitions_consumed"] = int(completion["total_simulator_transitions"])
        payload["training_protocol"] = completion["training_protocol"]
        payload["dense_finetune_transitions"] = local
        payload["initialization_simulator_transitions"] = int(completion["initialization_simulator_transitions"])
        payload["ppo_accounting_applicable"] = False
        payload["domain_label_available_to_actor"] = False
        payload["magnitude_threshold"] = float(completion["magnitude_threshold"])
        payload["evaluation_source_sha256"]["v39_protocol_adapter"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        original_atomic_json(payload, path)
    base.atomic_json = annotated


if __name__ == "__main__": install_protocol_adapter(); base.main()
