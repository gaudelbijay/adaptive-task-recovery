#!/usr/bin/env python3
"""Aggregate every matched router arm into one comparison table.

This reports the input-matched comparison: all arms consume the same
current-centered observation tensor, the same frozen specialists, and the same
evaluation seeds. The factorized sweep dispatch is reported as a separate
component rather than folded into the causal arm, because only that arm can
run it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

CONDITIONS = (
    "nominal", "ejection", "permanent_block", "temporary_block", "reverse_ejection",
)
Z = 1.959963985


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + Z * Z / n
    centre = p + Z * Z / (2 * n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Newcombe hybrid-score interval for a difference of two proportions."""
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    lower = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return p1 - p2, lower, upper


def collect(patterns: list[str]) -> dict:
    per = collections.defaultdict(lambda: {"episodes": 0, "safe": 0, "violations": 0})
    total = {"episodes": 0, "safe": 0, "violations": 0}
    files = sorted({f for pattern in patterns for f in glob.glob(pattern)})
    for path in files:
        record = json.loads(Path(path).read_text())
        condition = record["condition"]
        episodes = int(record["episodes"])
        safe = record.get("safe_successes")
        if safe is None:
            safe = int(round(record["safe_success_rate"] * episodes))
        violations = record.get("violations")
        if violations is None:
            violations = int(round(record.get("violation_rate", 0.0) * episodes))
        for bucket in (per[condition], total):
            bucket["episodes"] += episodes
            bucket["safe"] += int(safe)
            bucket["violations"] += int(violations)
    return {"per_condition": dict(per), "overall": total, "manifests": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="unstructured_gru")
    args = parser.parse_args()

    arms = {
        "causal_gru_with_dispatch": ["results/v10_confirmation_causal_gru_router*/*.json"],
        "causal_gru_matched": ["results/v10_dev_nodispatch_causal_router*/*.json"],
        "unstructured_gru": ["results/v10_confirmation_unstructured_gru_router*/*.json"],
        "static_mlp": ["results/v10_confirmation_static_mlp_router*/*.json"],
        "heuristic_v28": ["results/v10_dev_heuristic_v28/*.json"],
        "oracle_upper_bound": ["results/v10_dev_oracle_upper/*.json"],
    }
    results = {name: collect(patterns) for name, patterns in arms.items()}
    results = {k: v for k, v in results.items() if v["overall"]["episodes"] > 0}

    base = results.get(args.baseline)
    report = {"schema_version": 1, "seed_family": 347000000, "arms": {}}
    for name, data in results.items():
        overall = data["overall"]
        n, k = overall["episodes"], overall["safe"]
        entry = {
            "manifests": data["manifests"],
            "episodes": n,
            "safe_successes": k,
            "safe_success_rate": k / n,
            "safe_success_wilson_95": list(wilson(k, n)),
            "violations": overall["violations"],
            "violation_rate": overall["violations"] / n,
            "conditions": {},
        }
        for condition in CONDITIONS:
            bucket = data["per_condition"].get(condition)
            if not bucket:
                continue
            cn, ck = bucket["episodes"], bucket["safe"]
            entry["conditions"][condition] = {
                "episodes": cn,
                "safe_successes": ck,
                "safe_success_rate": ck / cn,
                "safe_success_wilson_95": list(wilson(ck, cn)),
                "violation_rate": bucket["violations"] / cn,
            }
        if base is not None and name != args.baseline:
            bn, bk = base["overall"]["episodes"], base["overall"]["safe"]
            gain, lo, hi = newcombe(k, n, bk, bn)
            entry["gain_over_baseline"] = {
                "baseline": args.baseline,
                "difference": gain,
                "newcombe_95": [lo, hi],
            }
            held = entry["conditions"].get("reverse_ejection")
            base_held = base["per_condition"].get("reverse_ejection")
            if held and base_held:
                g, l, h = newcombe(
                    held["safe_successes"], held["episodes"],
                    base_held["safe"], base_held["episodes"],
                )
                entry["heldout_reverse_gain"] = {
                    "difference": g, "newcombe_95": [l, h],
                }
        report["arms"][name] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    width = max(len(n) for n in report["arms"])
    header = f"{'arm':<{width}}  {'n':>5}  {'safe':>7}  {'viol':>6}  " + "  ".join(
        f"{c[:9]:>9}" for c in CONDITIONS
    )
    print(header)
    print("-" * len(header))
    for name, entry in report["arms"].items():
        cells = "  ".join(
            f"{entry['conditions'].get(c, {}).get('safe_success_rate', float('nan')):>9.4f}"
            for c in CONDITIONS
        )
        print(
            f"{name:<{width}}  {entry['episodes']:>5}  "
            f"{entry['safe_success_rate']:>7.4f}  {entry['violation_rate']:>6.4f}  {cells}"
        )
    print()
    for name, entry in report["arms"].items():
        gain = entry.get("gain_over_baseline")
        if not gain:
            continue
        lo, hi = gain["newcombe_95"]
        line = (
            f"  {name:<{width}} pooled {gain['difference']*100:+6.2f} pp "
            f"[{lo*100:+.2f}, {hi*100:+.2f}]"
        )
        held = entry.get("heldout_reverse_gain")
        if held:
            hlo, hhi = held["newcombe_95"]
            line += (
                f"   held-out reverse {held['difference']*100:+6.2f} pp "
                f"[{hlo*100:+.2f}, {hhi*100:+.2f}]"
            )
        print(line)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
