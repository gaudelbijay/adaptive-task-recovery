#!/usr/bin/env python3
"""Validate completed shards and export bootstrap statistics as CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from atr.evaluation.benchmark_suite import (
    aggregate_records,
    load_completed_records,
    load_spec,
    validate_result_completeness,
    write_summary_table_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-policy", default="oracle_feasibility")
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args()

    spec = load_spec(args.manifest)
    records = load_completed_records(args.run_dir)
    validate_result_completeness(spec, records)
    report = aggregate_records(
        records,
        metrics=spec.metrics,
        reference_policy=args.reference_policy,
        n_resamples=args.n_resamples,
        ci=args.ci,
    )
    run_dir = Path(args.run_dir)
    (run_dir / "aggregate.json").write_text(json.dumps(report, indent=2))
    write_summary_table_csv(report, run_dir / "summary_table.csv")
    print(json.dumps({
        "aggregate": str(run_dir / "aggregate.json"),
        "summary_table": str(run_dir / "summary_table.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
