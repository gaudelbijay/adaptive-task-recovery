#!/usr/bin/env python3
"""Fail-closed allocation gate for V35 translation repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from check_v31_multicamera_dagger_smoke_gate import check


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = check(args.config)
    payload["protocol"] = "V35 one-seed learned-translation allocation gate"
    payload["source_sha256"]["checker"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload["claim_boundary"] = json.loads(args.config.read_text())["claim_boundary"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["eligible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
