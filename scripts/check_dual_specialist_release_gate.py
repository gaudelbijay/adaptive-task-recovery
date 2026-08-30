#!/usr/bin/env python3
"""Release V19/V20 only when both integrated-state gates ran and failed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def load(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def check(config):
    primary_path = Path(config["primary_gate"])
    fallback_path = Path(config["fallback_gate"])
    primary, primary_hash = load(primary_path)
    fallback, fallback_hash = load(fallback_path)
    protocol = config["expected_protocol"]
    for payload, method, label in (
        (primary, config["primary_method"], "primary"),
        (fallback, config["fallback_method"], "fallback"),
    ):
        if payload.get("protocol") != protocol:
            raise ValueError(f"{label} gate has the wrong protocol")
        if payload.get("method") != method:
            raise ValueError(f"{label} gate has the wrong method")
        if not isinstance(payload.get("passed"), bool):
            raise ValueError(f"{label} gate lacks a boolean verdict")
    eligible = not primary["passed"] and not fallback["passed"]
    return {
        "schema_version": 1,
        "protocol": "fail-closed dual-specialist RGB allocation router",
        "primary_gate_passed": primary["passed"],
        "fallback_gate_passed": fallback["passed"],
        "eligible": eligible,
        "checks": {
            "primary_gate_ran_and_failed": not primary["passed"],
            "fallback_gate_ran_and_failed": not fallback["passed"],
        },
        "source_sha256": {
            str(primary_path): primary_hash,
            str(fallback_path): fallback_hash,
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
