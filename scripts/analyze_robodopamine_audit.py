#!/usr/bin/env python3
"""Analyse the Robo-Dopamine history-conditioning audit.

The claim under test is that history conditioning -- a REFERENCE START/END
panel supplied alongside the queried BEFORE/AFTER sets -- is what lifts the
model above "static before-after observations". The matched control is the same
released model on the same episodes with that panel removed.

Two things decide the verdict, in this order:

1. Competence. VOC is trivially saturable: their progress accumulator is
   monotone in the sign of the score, so a model that ignores the images and
   emits a constant positive score earns VOC = +1.0 on the forward direction,
   and a constant negative score earns +1.0 on the inverse direction. A
   fixed-sign predictor therefore averages ~0.5 across the two directions
   without perceiving anything. Unless both conditions clear that floor by a
   wide margin, no comparison between them means anything.

2. Match. Only if competence holds does the paired difference get a reading. A
   lower rung "matches" when the 95% bootstrap interval on the paired
   difference contains zero. Episodes are the resampling unit because steps
   within an episode share a progress curve; domains are reported separately
   because they are the family axis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "rd_audit", Path(__file__).resolve().parent / "audit_robodopamine_history.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)


def safe_voc(progress) -> float:
    """Their caller maps a nan/out-of-range correlation to 0.0."""
    v = rd.voc(progress)
    if not np.isfinite(v) or not (-1.0 <= v <= 1.0):
        return 0.0
    return v


def trivial_baselines(step_counts, seed=0):
    """VOC for image-blind predictors on the same episode lengths.

    constant_positive / constant_negative are the sign-bias exploits; random is
    a coin flip per step. Each is scored in both directions, exactly as the real
    conditions are.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("constant_positive", "constant_negative", "random_sign"):
        per_dir = {}
        for inverse in (False, True):
            vals = []
            for n in step_counts:
                if name == "constant_positive":
                    scores = [0.5] * n
                elif name == "constant_negative":
                    scores = [-0.5] * n
                else:
                    scores = list(rng.choice([-0.5, 0.5], size=n))
                vals.append(safe_voc(rd.progress_curve(scores, inverse)))
            per_dir["inverse" if inverse else "forward"] = float(np.mean(vals))
        per_dir["both"] = float(np.mean(list(per_dir.values())))
        out[name] = per_dir
    return out


def paired_bootstrap(pairs, n_boot=10000, seed=0):
    """95% interval on the mean paired difference, resampling episodes."""
    if not pairs:
        return float("nan"), (float("nan"), float("nan"))
    diffs = np.array([f - a for f, a in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return float(diffs.mean()), (float(np.percentile(boot, 2.5)),
                                 float(np.percentile(boot, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", default="results/robodopamine/audit_summary.json")
    ap.add_argument("--competence-margin", type=float, default=0.20,
                    help="Required margin over the strongest trivial predictor.")
    ap.add_argument("--ceiling", type=float, default=0.99,
                    help="VOC at or above which a pair carries no information.")
    ap.add_argument("--ceiling-fraction", type=float, default=0.90,
                    help="Domain is uninformative when this fraction is at ceiling.")
    ap.add_argument("--negligible", type=float, default=0.01,
                    help="Effect sizes below this are reported as negligible.")
    args = ap.parse_args()

    records = []
    for path in args.inputs:
        records.extend(json.loads(Path(path).read_text())["records"])
    if not records:
        raise SystemExit("no records")

    failures = sum(r.get("parse_failures", 0) for r in records)
    steps = [r["steps"] for r in records]

    # Pair full against ablated on the identical (domain, episode, direction).
    keyed = {(r["domain"], r["episode"], r["inverse"], r["condition"]): r for r in records}
    by_domain = defaultdict(list)
    score_deltas = defaultdict(list)
    for (dom, ep, inv, cond), r in keyed.items():
        if cond != "full":
            continue
        other = keyed.get((dom, ep, inv, "ablated"))
        if other is None:
            continue
        by_domain[dom].append((inv, safe_voc(r["progress"]), safe_voc(other["progress"])))
        # VOC is a rank statistic over few steps, so it quantises: two conditions
        # can differ in every raw score and still tie on VOC. The mean absolute
        # per-step score gap is the finer instrument, reported alongside.
        a, b = r.get("scores") or [], other.get("scores") or []
        if a and len(a) == len(b):
            score_deltas[dom].append(float(np.mean(np.abs(np.array(a) - np.array(b)))))

    trivial = trivial_baselines(steps)
    floor = max(t["both"] for t in trivial.values())

    all_pairs = [(f, a) for v in by_domain.values() for _, f, a in v]
    full_mean = float(np.mean([f for f, _ in all_pairs]))
    abl_mean = float(np.mean([a for _, a in all_pairs]))
    competent = (full_mean - floor > args.competence_margin
                 and abl_mean - floor > args.competence_margin)

    print(f"records={len(records)}  parse_failures={failures}  "
          f"episodes_paired={len(all_pairs)}  steps med={int(np.median(steps))}")
    print("\ntrivial predictors (image-blind), mean VOC:")
    for name, d in trivial.items():
        print(f"  {name:<18} forward={d['forward']:+.3f}  inverse={d['inverse']:+.3f}  "
              f"both={d['both']:+.3f}")
    print(f"\ncompetence floor (strongest trivial) = {floor:+.3f}")
    print(f"  full    mean VOC = {full_mean:+.3f}   margin = {full_mean - floor:+.3f}")
    print(f"  ablated mean VOC = {abl_mean:+.3f}   margin = {abl_mean - floor:+.3f}")
    print(f"  competence {'HOLDS' if competent else 'FAILS'} "
          f"(required margin {args.competence_margin:+.2f})")

    print("\nper-domain paired difference (full - ablated):")
    print(f"  {'domain':<16} {'n':>4} {'full':>7} {'abl':>7} {'diff':>9} "
          f"{'95% CI':>20} {'ceil':>6}  reading")
    domain_rows = {}
    for dom in sorted(by_domain):
        pairs = [(f, a) for _, f, a in by_domain[dom]]
        mean, (lo, hi) = paired_bootstrap(pairs)
        match = lo <= 0.0 <= hi
        # A pair is at ceiling when both conditions are pinned near the maximum:
        # nothing can be resolved there regardless of what the panel does.
        ceil = float(np.mean([1.0 if (f >= args.ceiling and a >= args.ceiling) else 0.0
                              for f, a in pairs]))
        delta = float(np.mean(score_deltas[dom])) if score_deltas[dom] else float("nan")
        if ceil >= args.ceiling_fraction:
            reading = "uninformative (at ceiling)"
        elif match:
            reading = "match"
        elif abs(mean) < args.negligible:
            reading = f"significant but negligible (<{args.negligible})"
        else:
            reading = "PANEL HELPS"
        domain_rows[dom] = {"n": len(pairs), "full": float(np.mean([f for f, _ in pairs])),
                            "ablated": float(np.mean([a for _, a in pairs])),
                            "diff": mean, "ci": [lo, hi], "matches": bool(match),
                            "ceiling_fraction": ceil, "mean_abs_score_delta": delta,
                            "reading": reading}
        print(f"  {dom:<16} {len(pairs):>4} {domain_rows[dom]['full']:>+7.3f} "
              f"{domain_rows[dom]['ablated']:>+7.3f} {mean:>+9.4f} "
              f"[{lo:+.4f},{hi:+.4f}] {ceil:>6.2f}  {reading}")
    print("\n  mean |per-step score gap| (finer than VOC, which quantises):")
    for dom in sorted(score_deltas):
        print(f"    {dom:<16} {np.mean(score_deltas[dom]):.4f}")

    # The domains with headroom are the ones that can actually answer the
    # question, so they get their own pooled reading.
    head_pairs = [(f, a) for dom in by_domain for _, f, a in by_domain[dom]
                  if not (f >= args.ceiling and a >= args.ceiling)]
    hmean, (hlo, hhi) = paired_bootstrap(head_pairs)
    print(f"\nheadroom-only (excluding pairs pinned >= {args.ceiling}): "
          f"n={len(head_pairs)}  diff={hmean:+.4f} [{hlo:+.4f}, {hhi:+.4f}]  "
          f"{'match' if hlo <= 0 <= hhi else 'PANEL HELPS'}")

    mean, (lo, hi) = paired_bootstrap(all_pairs)
    pooled_match = lo <= 0.0 <= hi
    print(f"\npooled  diff={mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
          f"{'match' if pooled_match else 'PANEL HELPS'}")

    if not competent:
        verdict = ("inconclusive: neither condition clears the trivial-predictor "
                   "floor by the required margin")
    elif pooled_match:
        verdict = ("the reference panel is not necessary: removing it leaves VOC "
                   "statistically unchanged while both conditions remain far above "
                   "the trivial floor")
    else:
        verdict = ("the reference panel carries measurable weight: removing it "
                   "costs VOC beyond sampling error")
    print(f"\nverdict: {verdict}")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": 1,
        "claim_under_test": (
            "Robo-Dopamine 2.0 attributes its gain to history conditioning via a "
            "reference panel, against static before-after baselines."),
        "n_records": len(records), "parse_failures": failures,
        "episodes_paired": len(all_pairs),
        "trivial_baselines": trivial, "competence_floor": floor,
        "competence_margin_required": args.competence_margin,
        "full_mean_voc": full_mean, "ablated_mean_voc": abl_mean,
        "competence_holds": bool(competent),
        "per_domain": domain_rows,
        "ceiling_threshold": args.ceiling,
        "headroom_only": {"n": len(head_pairs), "diff": hmean, "ci": [hlo, hhi],
                          "matches": bool(hlo <= 0 <= hhi)},
        "pooled": {"diff": mean, "ci": [lo, hi], "matches": bool(pooled_match)},
        "verdict": verdict,
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
