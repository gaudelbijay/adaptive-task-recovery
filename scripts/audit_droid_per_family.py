#!/usr/bin/env python3
"""Per-family ladder verdicts on the DROID success benchmark.

This is the second external test of the paper's headline claim: that a pooled
verdict averages over families that disagree, so a benchmark can clear the audit
overall while individual held-out families do not support it.

It reuses the REBOOT per-family logic exactly -- the same criterion, the same
resampling unit, the same reproduction guard against the pooled figures -- so
the two external benchmarks are scored identically and any difference between
them is a property of the data rather than of the analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RUNG4 = "causal_dynamics_gru"
LOWER = {"static_mlp": 1, "endpoint_pair_mlp": 2, "moment_mlp": 2.5}
POOLED_COMPARATOR = "moment_mlp"


def bootstrap_over_seeds(differences, seed, samples):
    """Percentile interval on the mean paired difference, resampling seeds.

    Within one family every rung is scored on the identical held-out episodes,
    so the difference is paired and optimizer randomness is the only remaining
    variation.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True)
    return np.percentile(draws.mean(axis=1), [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-glob", default="results/droid/droid_ladder_seed*.json")
    parser.add_argument("--output", default="results/droid/droid_per_family_v1.json")
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.seed_glob))
    if not paths:
        raise SystemExit(f"no per-seed records matched {args.seed_glob}")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    families = [f["test_object"] for f in runs[0]["folds"]]
    for run in runs:
        if [f["test_object"] for f in run["folds"]] != families:
            raise SystemExit("fold order differs between seeds; refusing to pool")

    def auroc(index, method):
        return np.array([r["folds"][index][method]["auroc"] for r in runs])

    pooled = {m: float(np.mean([auroc(i, m).mean() for i in range(len(families))]))
              for m in [RUNG4, *LOWER]}

    results, matched = [], []
    for index, family in enumerate(families):
        scores = {m: auroc(index, m) for m in [RUNG4, *LOWER]}
        best = max(LOWER, key=lambda m: scores[m].mean())
        difference = scores[RUNG4] - scores[best]
        low, high = bootstrap_over_seeds(difference, args.bootstrap_seed + index, args.samples)
        match = bool(low <= 0.0 <= high)
        matched.append(match)
        results.append({
            "family": family,
            "rung_means": {m: float(scores[m].mean()) for m in scores},
            "best_lower_rung": best,
            "difference": float(difference.mean()),
            "seed_bootstrap_95": [float(low), float(high)],
            "matches_rung4": match,
        })

    report = {
        "schema_version": 1,
        "benchmark": "DROID-success",
        "protocol": "per-family ladder verdict, leave-one-building-out",
        "resampling_unit": "optimizer seed within family",
        "claim_boundary": (
            "A match means the recurrent model is not distinguishable from the "
            "best lower rung under optimizer noise. It is not evidence that no "
            "difference exists."
        ),
        "optimizer_seeds": len(runs),
        "families": len(families),
        "families_matched": int(sum(matched)),
        "pooled_rung_means": pooled,
        "pooled_difference_vs_order_free": pooled[RUNG4] - pooled[POOLED_COMPARATOR],
        "per_family": results,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'family':<26}{'best lower':>18}{'R4':>8}{'diff':>9}   95% CI")
    for row in results:
        lo, hi = row["seed_bootstrap_95"]
        flag = "  match" if row["matches_rung4"] else ""
        print(f"{row['family'][:25]:<26}{row['best_lower_rung']:>18}"
              f"{row['rung_means'][RUNG4]:8.4f}{row['difference']:+9.4f}"
              f"   [{lo:+.4f}, {hi:+.4f}]{flag}")
    print(f"\n{report['families_matched']} of {report['families']} families matched; "
          f"pooled difference {report['pooled_difference_vs_order_free']:+.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
