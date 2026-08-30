#!/usr/bin/env python3
"""Resolve the V22 gate after any smoke outcome and route failure-only work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from check_drac_stability_smoke_gate import check


def resolve(config_path: str | Path) -> dict:
    path = Path(config_path)
    raw = path.read_bytes()
    try:
        payload = check(json.loads(raw))
        payload["resolution"] = "eligible" if payload["eligible"] else "ineligible"
    except Exception as error:  # fail closed for missing/crashed/inexact smoke
        payload = {
            "schema_version": 1,
            "protocol": "V22 failure-only fallback router",
            "eligible": False,
            "resolution": "gate_error",
            "error_type": type(error).__name__,
            "error": str(error),
            "claim_boundary": (
                "Routing evidence only. A gate error may release the short V23 "
                "runtime pilot, but cannot support a V22 or V23 performance claim."
            ),
        }
    payload["config"] = str(path)
    payload["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return payload


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = resolve(args.config)
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
