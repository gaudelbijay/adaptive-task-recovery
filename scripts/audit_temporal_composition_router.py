#!/usr/bin/env python3
"""Audit the pre-closed-loop gates for the V2 temporal-composition router."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, default=Path(
        "configs/a_plus_recovery_gate_v3_full_geometry.json"
    ))
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-reject", action="store_true")
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    paths = sorted(args.summary_dir.glob("summary_seed*.json"))
    expected_seeds = gate["pass_criteria"]["minimum_independent_training_seeds"]
    if len(paths) != expected_seeds:
        raise RuntimeError(f"expected {expected_seeds} seed summaries, found {len(paths)}")
    summaries = [json.loads(path.read_text()) for path in paths]
    methods = {}
    for name in ("causal_gru", "static_mlp", "unstructured_gru"):
        heldout = [run["models"][name]["test"]["heldout_option_accuracy"] for run in summaries]
        observed = [run["models"][name]["test"]["accuracy"] for run in summaries]
        methods[name] = {
            "heldout_option_accuracy_by_seed": heldout,
            "heldout_option_accuracy_mean": sum(heldout) / len(heldout),
            "all_option_accuracy_by_seed": observed,
            "all_option_accuracy_mean": sum(observed) / len(observed),
        }
    criteria = gate["pass_criteria"]
    causal_min = criteria.get(
        "causal_heldout_reverse_offline_accuracy_min",
        criteria.get("causal_heldout_offline_accuracy_min"),
    )
    static_max = criteria.get(
        "static_heldout_reverse_offline_accuracy_max",
        criteria.get("static_heldout_offline_accuracy_max"),
    )
    if causal_min is None or static_max is None:
        raise RuntimeError("gate omits held-out causal/static offline thresholds")
    checks = {
        "training_seed_count": len(paths) >= expected_seeds,
        "causal_heldout_reverse_accuracy": (
            methods["causal_gru"]["heldout_option_accuracy_mean"]
            >= causal_min
        ),
        "static_heldout_reverse_shortcut_absent": (
            methods["static_mlp"]["heldout_option_accuracy_mean"]
            <= static_max
        ),
    }
    payload = {
        "schema_version": 1,
        "gate": str(args.gate),
        "summaries": [str(path) for path in paths],
        "methods": methods,
        "checks": checks,
        "offline_gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.fail_on_reject and not payload["offline_gate_pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
