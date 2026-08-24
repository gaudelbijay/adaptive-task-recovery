#!/usr/bin/env python3
"""Run or preflight one shard of a versioned benchmark manifest."""

from __future__ import annotations

import argparse
import json
import os

from atr.evaluation.benchmark_suite import (
    expand_cases,
    load_spec,
    pilot_spec,
    run_shard,
    shard_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="results/benchmarks")
    parser.add_argument(
        "--shard-index", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument(
        "--shard-count", type=int,
        default=int(os.environ.get("ATR_SHARD_COUNT", "1")),
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--pilot", action="store_true",
        help="run one seed in every matrix cell under a distinct fingerprint",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.manifest)
    if args.pilot:
        spec = pilot_spec(spec)
    cases = expand_cases(spec)
    selected = shard_cases(cases, args.shard_index, args.shard_count)
    if args.preflight:
        print(json.dumps({
            "name": spec.name,
            "fingerprint": spec.fingerprint,
            "total_cases": len(cases),
            "policies": list(spec.policies),
            "total_policy_runs": len(cases) * len(spec.policies),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "cases_in_shard": len(selected),
            "policy_runs_in_shard": len(selected) * len(spec.policies),
        }, indent=2))
        return
    print(json.dumps(run_shard(
        spec,
        args.output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        fail_fast=args.fail_fast,
    ), indent=2))


if __name__ == "__main__":
    main()
