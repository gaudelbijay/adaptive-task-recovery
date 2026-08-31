#!/usr/bin/env python3
"""Aggregate preregistered OOD router manifests without hiding weak axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_axis(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("axis must be NAME=DIRECTORY")
    return name, Path(directory)


def aggregate(directory: Path, expected_manifests: int) -> dict:
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if len(records) != expected_manifests:
        raise RuntimeError(
            f"{directory}: expected {expected_manifests} manifests, found {len(records)}"
        )
    episodes = sum(int(row["episodes"]) for row in records)
    safe = sum(int(row["safe_successes"]) for row in records)
    violations = sum(int(row["violations"]) for row in records)
    return {
        "manifests": len(records),
        "episodes": episodes,
        "safe_successes": safe,
        "safe_success_rate": safe / episodes,
        "violations": violations,
        "violation_rate": violations / episodes,
        "conditions": {
            condition: {
                "episodes": sum(int(row["episodes"]) for row in records if row["condition"] == condition),
                "safe_successes": sum(int(row["safe_successes"]) for row in records if row["condition"] == condition),
            }
            for condition in sorted({row["condition"] for row in records})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--axis", action="append", type=parse_axis, required=True)
    parser.add_argument("--expected-manifests-per-axis", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    axes = {
        name: aggregate(directory, args.expected_manifests_per_axis)
        for name, directory in args.axis
    }
    episodes = sum(row["episodes"] for row in axes.values())
    safe = sum(row["safe_successes"] for row in axes.values())
    violations = sum(row["violations"] for row in axes.values())
    threshold = float(gate["pass_criteria"]["ood_safe_success_min"])
    result = {
        "schema_version": 1,
        "gate": str(args.gate),
        "selection_seed_base": gate["selection_seed_base"],
        "axes": axes,
        "pooled": {
            "episodes": episodes,
            "safe_successes": safe,
            "safe_success_rate": safe / episodes,
            "violations": violations,
            "violation_rate": violations / episodes,
        },
        "threshold": threshold,
        "ood_gate_pass": safe / episodes >= threshold,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
