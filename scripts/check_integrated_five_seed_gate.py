#!/usr/bin/env python3
"""Fail closed unless the frozen integrated V13 screen authorizes new seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config, config_hash = load(config_path)
    selection_path = Path(config["selection_artifact"])
    selection, selection_hash = load(selection_path)
    screening = list(config["screening_seeds"])
    confirmatory = list(config["confirmatory_seeds"])
    if len(screening) != 3 or len(confirmatory) != 2:
        raise ValueError("confirmation requires exactly three screen and two new seeds")
    if len(set(screening + confirmatory)) != 5:
        raise ValueError("screening and confirmatory seeds must be unique")
    if selection.get("protocol") != "predeclared integrated visual-policy selection":
        raise ValueError("selection artifact has the wrong protocol")
    if selection.get("thresholds") != config["expected_thresholds"]:
        raise ValueError("selection artifact changed a frozen eligibility threshold")
    expected = config["expected_selected"]
    selected = selection.get("selected")
    if selected != expected or selection.get("all_candidates_ineligible"):
        raise RuntimeError(
            f"confirmatory allocation rejected: expected {expected!r}, selected {selected!r}"
        )
    candidate = next(
        (item for item in selection.get("candidates", []) if item.get("label") == expected),
        None,
    )
    if candidate is None or not candidate.get("eligible"):
        raise RuntimeError("selected candidate lacks complete eligibility evidence")
    checks = candidate.get("checks")
    if set(checks or {}) != set(config["expected_checks"]):
        raise RuntimeError("selected candidate has missing or unexpected checks")
    if not all(checks.values()):
        raise RuntimeError("selected candidate did not pass every frozen check")
    config_hashes = {}
    training_configs = config.get("training_configs")
    if training_configs is None:
        training_configs = [config[key] for key in (
            "visual_clean_append_config", "visual_integrated_append_config",
            "state_integrated_append_config",
        )]
    if not isinstance(training_configs, list) or not training_configs:
        raise ValueError("confirmation requires at least one frozen training config")
    if len(set(training_configs)) != len(training_configs):
        raise ValueError("confirmatory training configs must be unique")
    for path in training_configs:
        payload, digest = load(Path(path))
        if payload.get("seeds") != confirmatory:
            raise ValueError("confirmatory config has wrong untouched seeds")
        config_hashes[path] = digest
    payload = {
        "schema_version": 1,
        "protocol": "integrated five-seed allocation gate",
        "authorized": True,
        "selected": selected,
        "screening_seeds": screening,
        "confirmatory_seeds": confirmatory,
        "selection_artifact": str(selection_path),
        "selection_sha256": selection_hash,
        "config": str(config_path),
        "config_sha256": config_hash,
        "training_config_sha256": config_hashes,
        "claim_boundary": config["claim_boundary"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
