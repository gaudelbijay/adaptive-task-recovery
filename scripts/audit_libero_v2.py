#!/usr/bin/env python3
"""Does LIBERO require its language instruction at decision time?

Each LIBERO suite pairs a scene with an instruction. If a control that never
sees the instruction can name the task from what a policy observes *before
acting*, then a policy can succeed on that suite without grounding language,
and the suite does not test what it is usually taken to test.

Rungs, all predicting the 10-way task label:
  trivial    chance, 1/10
  duration   demonstration length alone, no vision at all
  R1         the initial agentview frame at t=0, which is what a policy sees
             before its first action
  R2b        an order-free mean/std summary over an early prefix. Reported for
             completeness but NOT read as a policy shortcut: these are expert
             frames, and a policy must generate that motion rather than observe
             it. It bounds task inference from a prefix, not policy execution.

Statistics: accuracy is averaged over repeated stratified CV with several
seeds, so the interval reflects split variance as well as item variance.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]


def repeated_cv(X, y, seeds=(0, 1, 2, 3, 4), folds=5):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    per_seed, correct_any = [], []
    for sd in seeds:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=sd)
        preds = np.zeros(len(y), dtype=int)
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=1.0))
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict(X[te])
        correct = (preds == y).astype(float)
        per_seed.append(correct.mean())
        correct_any.append(correct)
    return np.array(per_seed), np.stack(correct_any)


def boot_ci(mat, n_boot=10000, seed=0):
    """Resample demos; average over seeds within each resample."""
    rng = np.random.default_rng(seed)
    n = mat.shape[1]
    idx = rng.integers(0, n, size=(n_boot, n))
    b = mat[:, idx].mean(axis=0).mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.expanduser("~/atr-libero/features"))
    ap.add_argument("--out", default=os.path.expanduser("~/atr-libero/libero_audit.json"))
    args = ap.parse_args()

    report = {}
    for suite in SUITES:
        path = os.path.join(args.features, f"{suite}.npz")
        if not os.path.exists(path):
            print(f"  (skipping {suite}: features not built)"); continue
        d = np.load(path, allow_pickle=True)
        y = d["label"]
        chance = 1.0 / len(np.unique(y))
        rungs = {"duration": d["length"].astype(float),
                 "R1_initial_frame": d["first"],
                 "R2b_order_free_prefix": d["summary"]}
        print(f"\n=== {suite}  ({len(y)} demos, {len(np.unique(y))} tasks, "
              f"chance {chance:.3f}) ===")
        row = {"n": int(len(y)), "n_tasks": int(len(np.unique(y))),
               "chance": chance, "rungs": {}}
        for name, X in rungs.items():
            per_seed, mat = repeated_cv(X, y)
            acc = float(mat.mean())
            lo, hi = boot_ci(mat)
            row["rungs"][name] = {
                "accuracy": acc, "ci": [lo, hi],
                "margin_over_chance": acc - chance,
                "seed_spread": [float(per_seed.min()), float(per_seed.max())]}
            print(f"  {name:<24} acc={acc:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"margin {acc - chance:+.3f}  seeds[{per_seed.min():.3f},"
                  f"{per_seed.max():.3f}]")
        r1 = row["rungs"]["R1_initial_frame"]
        row["language_required_at_decision_time"] = bool(r1["ci"][0] <= chance * 1.5)
        row["verdict"] = ("language is load-bearing: the task is not identifiable "
                          "before acting"
                          if row["language_required_at_decision_time"] else
                          "language is redundant: the task is identifiable from the "
                          "initial observation alone")
        print(f"  -> {row['verdict']}")
        report[suite] = row

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
