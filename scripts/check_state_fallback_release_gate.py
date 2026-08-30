#!/usr/bin/env python3
"""Release the state fallback only after an explicit primary-gate failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def check(config):
    gate_path = Path(config["primary_gate"])
    raw = gate_path.read_bytes()
    gate = json.loads(raw)
    if gate.get("protocol") != config["expected_protocol"]:
        raise ValueError("primary gate has the wrong protocol")
    if gate.get("method") != config["primary_method"]:
        raise ValueError("primary gate has the wrong method")
    if not isinstance(gate.get("passed"), bool):
        raise ValueError("primary gate lacks a boolean verdict")
    eligible = not gate["passed"]
    return {
        "schema_version": 1,
        "protocol": "fail-closed state-fallback allocation router",
        "primary_gate_passed": gate["passed"],
        "eligible": eligible,
        "checks": {"primary_gate_ran_and_failed": eligible},
        "source_sha256": {
            str(gate_path): hashlib.sha256(raw).hexdigest(),
        },
        "claim_boundary": config["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    payload = check(json.loads(config_bytes))
    payload["config"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
