#!/usr/bin/env python3
"""Fail closed unless the strict state teacher retains nominal competence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=768)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--minimum-raw", type=float, default=0.70)
    parser.add_argument("--minimum-safe", type=float, default=0.70)
    parser.add_argument("--maximum-violation", type=float, default=0.05)
    args = parser.parse_args()

    aggregate_path = Path(args.aggregate)
    aggregate_bytes = aggregate_path.read_bytes()
    aggregate = json.loads(aggregate_bytes)
    nominal = aggregate.get("nominal_condition")
    if not isinstance(nominal, list):
        raise ValueError("aggregate lacks a nominal_condition list")
    records = [item for item in nominal if item.get("method") == args.method]
    if len(records) != 1:
        raise ValueError("aggregate lacks exactly one requested nominal method")
    record = records[0]
    expected = {"episodes": args.episodes, "seeds": args.seeds}
    for key, value in expected.items():
        if int(record.get(key, -1)) != value:
            raise ValueError(
                f"nominal teacher {key} mismatch: expected {value}, "
                f"observed {record.get(key)!r}"
            )
    raw = float(record["pooled_success_rate"])
    safe = float(record["pooled_safe_success_rate"])
    violation = float(record["constraint_violation_rate"])
    checks = {
        "raw": raw >= args.minimum_raw,
        "safe": safe >= args.minimum_safe,
        "violation": violation <= args.maximum_violation,
    }
    payload = {
        "schema_version": 1,
        "protocol": "strict-state teacher nominal-competence gate",
        "aggregate": str(aggregate_path),
        "aggregate_sha256": hashlib.sha256(aggregate_bytes).hexdigest(),
        "method": args.method,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "raw_success_rate": raw,
        "safe_success_rate": safe,
        "constraint_violation_rate": violation,
        "thresholds": {
            "minimum_raw": args.minimum_raw,
            "minimum_safe": args.minimum_safe,
            "maximum_violation": args.maximum_violation,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "claim_boundary": (
            "training-allocation gate only; does not constitute a visual-policy "
            "result or alter preregistered hypotheses"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
