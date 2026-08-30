#!/usr/bin/env python3
"""Release new-seed confirmation only after the frozen V5 screen passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from check_visual_competence_gate import check_visualization_gate


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def confirmatory_gate(
    visual_payload, state_payload, visual_method, state_method_name,
    new_seeds, minimum_nominal=0.70,
):
    matches = [
        item for item in state_payload.get("environments", [])
        if item.get("method") == state_method_name
    ]
    if len(matches) != 1:
        raise ValueError("confirmatory gate lacks exactly one state reference")
    reference = matches[0]
    if int(reference.get("seeds", -1)) != 3 or int(reference.get("episodes", -1)) != 768:
        raise ValueError("confirmatory gate state reference has the wrong protocol")
    result = check_visualization_gate(
        visual_payload, visual_method,
        float(reference["pooled_success_rate"]),
        float(reference["pooled_safe_success_rate"]),
        float(reference["constraint_violation_rate"]),
        float(minimum_nominal), seeds=3, episodes=768,
    )
    if len(new_seeds) != 2 or len(set(new_seeds)) != 2:
        raise ValueError("confirmatory protocol requires exactly two distinct new seeds")
    if set(new_seeds) & {9351, 4796, 1788}:
        raise ValueError("confirmatory seeds overlap the screening seeds")
    result.update({
        "protocol": "three-seed V5 screen gates two predeclared new training seeds",
        "screening_seeds": [9351, 4796, 1788],
        "confirmatory_seeds": list(new_seeds),
        "seed_derivation": (
            "first two unique integers from numpy default_rng(20260828)."
            "integers(1,100000), excluding screening seeds"
        ),
        "state_reference_method": state_method_name,
        "no_confirmatory_seed_may_be_discarded": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-aggregate", required=True)
    parser.add_argument("--state-aggregate", required=True)
    parser.add_argument("--visual-method", required=True)
    parser.add_argument("--state-method", required=True)
    parser.add_argument("--new-seeds", nargs=2, type=int, required=True)
    parser.add_argument("--minimum-nominal", type=float, default=0.70)
    parser.add_argument(
        "--output", default="results/final_visual_comparison/confirmatory_gate.json",
    )
    args = parser.parse_args()
    result = confirmatory_gate(
        json.loads(Path(args.visual_aggregate).read_text(encoding="utf-8")),
        json.loads(Path(args.state_aggregate).read_text(encoding="utf-8")),
        args.visual_method, args.state_method, args.new_seeds, args.minimum_nominal,
    )
    atomic_json(result, Path(args.output))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
