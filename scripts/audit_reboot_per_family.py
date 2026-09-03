#!/usr/bin/env python3
"""Run the ladder verdict per held-out object family, not only in aggregate.

The pooled REBOOT verdict is a single number over nine leave-one-object-out
folds. That number can only say whether the recurrent model beats the strongest
lower rung *on average*. It cannot say whether the benchmark's individual
held-out families are each measuring what they claim, and averaging is exactly
the operation that would hide a family whose held-out object is shortcut-solved.

This re-runs the matching criterion inside each family. The replication unit is
the optimizer seed, so a "match" here means the recurrent model is not
distinguishable from the best lower rung given optimizer noise -- a weaker
statement than "no difference exists", and reported as such.

Input is the per-seed fold records written by evaluate_reboot_causal_prefix.py;
no model is retrained, so this cannot disagree with the pooled figures. The
script asserts that it reproduces them before reporting anything.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

RUNG4 = "causal_dynamics_gru"
LOWER = {"static_mlp": 1, "endpoint_pair_mlp": 2, "moment_mlp": 2.5}
# The pooled artifact compares rung 4 against the order-free summary, so that
# is the comparator to reproduce when checking this script against it.
POOLED_COMPARATOR = "moment_mlp"


def bootstrap_over_seeds(differences, seed, samples):
    """Percentile interval on the mean paired difference, resampling seeds.

    Within one family every rung is scored on the identical held-out episodes,
    so the difference is paired and the only remaining variation is optimizer
    randomness. That makes the seed the resampling unit here, unlike the pooled
    audit where whole object families are resampled.
    """
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(samples, len(differences)), replace=True)
    return np.percentile(draws.mean(axis=1), [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-glob", default="results/reboot/reboot_ladder_v5_tenseed_seed*.json")
    parser.add_argument("--pooled", default="results/a_plus_audit/reboot_ladder_v5_aggregate.json")
    parser.add_argument("--output", default="results/a_plus_audit/reboot_per_family_v1.json")
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.seed_glob))
    if not paths:
        raise SystemExit(f"no per-seed records matched {args.seed_glob}")
    runs = [json.loads(Path(p).read_text()) for p in paths]
    families = [fold["test_object"] for fold in runs[0]["folds"]]
    for run in runs:
        if [f["test_object"] for f in run["folds"]] != families:
            raise SystemExit("fold order differs between seeds; refusing to pool")

    def auroc(family_index, method):
        return np.array([r["folds"][family_index][method]["auroc"] for r in runs])

    # Guard: this analysis must not silently disagree with the frozen pooled run.
    pooled = json.loads(Path(args.pooled).read_text())
    for method in [RUNG4, POOLED_COMPARATOR, *LOWER]:
        recomputed = float(np.mean([auroc(i, method).mean() for i in range(len(families))]))
        published = pooled["aggregate"][method]["macro_auroc_mean"]
        if abs(recomputed - published) > 1e-6:
            raise SystemExit(
                f"{method}: recomputed {recomputed:.6f} != published {published:.6f}"
            )

    results, matched = [], []
    for index, family in enumerate(families):
        scores = {m: auroc(index, m) for m in [RUNG4, *LOWER]}
        # The ladder always nominates the strongest lower rung, which need not be
        # the same rung in every family -- and is not, here.
        best = max(LOWER, key=lambda m: scores[m].mean())
        difference = scores[RUNG4] - scores[best]
        low, high = bootstrap_over_seeds(difference, args.bootstrap_seed + index, args.samples)
        match = bool(low <= 0.0 <= high)
        matched.append(match)
        results.append({
            "family": family,
            "rung_means": {m: float(scores[m].mean()) for m in scores},
            "best_lower_rung": best,
            "best_lower_rung_number": LOWER[best],
            "difference": float(difference.mean()),
            "seed_bootstrap_95": [float(low), float(high)],
            "matches_rung4": match,
        })

    report = {
        "schema_version": 1,
        "protocol": "per-family ladder verdict, leave-one-object-out",
        "resampling_unit": "optimizer seed within family",
        "claim_boundary": (
            "A match means the recurrent model is not distinguishable from the "
            "best lower rung under optimizer noise across ten seeds. It is not "
            "evidence that no difference exists."
        ),
        "optimizer_seeds": len(runs),
        "families": len(families),
        "families_matched": int(sum(matched)),
        "pooled_difference_vs_order_free": pooled["comparisons"]["causal_vs_moment_mlp"][
            "macro_auroc_difference"],
        "per_family": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'family':<15}{'best lower':>18}{'R4':>8}{'diff':>9}   95% CI")
    for row in results:
        flag = "  match" if row["matches_rung4"] else ""
        low, high = row["seed_bootstrap_95"]
        print(f"{row['family']:<15}{row['best_lower_rung']:>18}"
              f"{row['rung_means'][RUNG4]:8.4f}{row['difference']:+9.4f}"
              f"   [{low:+.4f}, {high:+.4f}]{flag}")
    print(f"\n{report['families_matched']} of {report['families']} families matched; "
          f"pooled difference {report['pooled_difference_vs_order_free']:+.4f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
