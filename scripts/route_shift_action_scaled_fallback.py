#!/usr/bin/env python3
"""Authorize V25 only from an explicit, valid V24 gate rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


PROTOCOL = (
    "pre-held-out matched-budget bounded shift-action stability allocation gate"
)
EXPECTED_CHECKS = {
    "best_success_at_end",
    "best_constraint_violation",
    "best_score_margin",
    "tail_mean_constraint_violation",
    "tail_mean_score_improvement",
    "finite_bounded_consistency",
}
EXPECTED_SCHEDULED_STEP = 19_996_672


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(result_path: str | Path, gate_config_path: str | Path) -> dict:
    result_path = Path(result_path)
    gate_config_path = Path(gate_config_path)
    base = {
        "schema_version": 1,
        "protocol": "V24-to-V25 explicit-rejection fallback router",
        "upstream_result": str(result_path),
        "gate_config": str(gate_config_path),
        "gate_config_sha256": sha256(gate_config_path),
        "authorize_v25": False,
        "claim_boundary": (
            "Routing evidence only. Authorization allocates a post-hoc smoke "
            "test and is not performance, robustness, or held-out evidence."
        ),
    }
    try:
        raw = result_path.read_bytes()
        upstream = json.loads(raw)
        if upstream.get("protocol") != PROTOCOL:
            raise ValueError("V24 gate artifact has the wrong protocol")
        if upstream.get("config_sha256") != base["gate_config_sha256"]:
            raise ValueError("V24 gate artifact does not match the frozen config")
        if int(upstream.get("scheduled_step", -1)) != EXPECTED_SCHEDULED_STEP:
            raise ValueError("V24 gate artifact has the wrong scheduled budget")
        checks = upstream.get("checks")
        if not isinstance(checks, dict) or set(checks) != EXPECTED_CHECKS:
            raise ValueError("V24 gate artifact has incomplete checks")
        if any(type(value) is not bool for value in checks.values()):
            raise ValueError("V24 gate checks are not Boolean")
        eligible = upstream.get("eligible")
        if type(eligible) is not bool or eligible is not all(checks.values()):
            raise ValueError("V24 eligibility disagrees with its checks")
        if not isinstance(upstream.get("candidate_training_source_sha256"), dict):
            raise ValueError("V24 gate lacks candidate training provenance")
        checkpoint_hash = upstream.get("candidate_best_checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
            raise ValueError("V24 gate lacks candidate checkpoint provenance")
        base.update({
            "upstream_result_sha256": hashlib.sha256(raw).hexdigest(),
            "upstream_eligible": eligible,
            "upstream_checks": checks,
            "resolution": "v24_eligible" if eligible else "v24_explicitly_rejected",
            "authorize_v25": not eligible,
        })
    except Exception as error:
        base.update({
            "resolution": "invalid_or_missing_v24_gate",
            "error_type": type(error).__name__,
            "error": str(error),
        })
    return base


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-result", required=True)
    parser.add_argument("--gate-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = resolve(args.upstream_result, args.gate_config)
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["authorize_v25"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
