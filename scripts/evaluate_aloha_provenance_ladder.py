#!/usr/bin/env python3
"""Run the ladder on the ALOHA provenance benchmark.

This is a second externally built benchmark for the audit. The claim under test
is that identifying demonstration provenance -- human teleoperation versus a
scripted policy -- on a task the model never trained on requires temporal
structure. The ladder asks whether a control lacking that structure does just as
well.

The REBOOT evaluator reserves a whole family for validation, which needs at
least three. This benchmark has two tasks, so validation is instead a
held-out slice of the *training* task's episodes, chosen so no episode appears
in both training and validation. Everything else -- the models, the fitting
routine, the metrics, the per-(method, fold) seeding -- is imported from that
evaluator unchanged, so the two benchmarks are scored by the same code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_reboot_causal_prefix import (  # noqa: E402
    fit_one, metrics, predict,
)

METHODS = ("static_mlp", "endpoint_pair_mlp", "moment_mlp",
           "unstructured_gru", "causal_dynamics_gru")
RUNG = {"static_mlp": 1, "endpoint_pair_mlp": 2, "moment_mlp": 2.5,
        "causal_dynamics_gru": 4, "unstructured_gru": 4}
TOP = "causal_dynamics_gru"


def episode_bootstrap(top_correct, low_correct, seed=20260903, samples=20000):
    """Paired interval on the difference, resampling whole episodes.

    Both rungs score the identical episodes, so the difference is paired and the
    episode is the resampling unit -- the same convention the simulated
    benchmarks use.
    """
    rng = np.random.default_rng(seed)
    difference = top_correct - low_correct
    draws = rng.choice(difference, size=(samples, len(difference)), replace=True)
    return float(difference.mean()), np.percentile(draws.mean(axis=1), [2.5, 97.5])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/aloha/aloha_provenance_v1.npz")
    parser.add_argument("--audit", default="results/aloha/aloha_provenance_v1.audit.json")
    parser.add_argument("--output", default="results/aloha/aloha_ladder_seed0.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    args = parser.parse_args()
    args.train_fraction = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data)
    sequence, label = raw["sequence"], raw["label"]
    family, episode = raw["object_id"], raw["episode_id"]
    audit = json.loads(Path(args.audit).read_text())
    names = {index: name for name, index in audit["families"].items()}

    folds = []
    for test_family in sorted(np.unique(family)):
        test_mask = family == test_family
        pool = np.flatnonzero(~test_mask)
        # Validation is a held-out slice of the training task, stratified by
        # label so both provenances are represented.
        rng = np.random.default_rng(args.seed * 977 + int(test_family))
        validation = []
        for value in np.unique(label[pool]):
            rows = pool[label[pool] == value]
            rng.shuffle(rows)
            validation.extend(rows[:max(1, int(len(rows) * args.validation_fraction))])
        validation_mask = np.zeros(len(label), dtype=bool)
        validation_mask[np.asarray(validation, dtype=int)] = True
        train_mask = (~test_mask) & (~validation_mask)

        fold = {"test_family": names[int(test_family)],
                "train_episodes": int(train_mask.sum()),
                "validation_episodes": int(validation_mask.sum()),
                "test_episodes": int(test_mask.sum())}
        test_index = np.flatnonzero(test_mask)
        horizon = max(args.horizons)
        for name in METHODS:
            model, mean, scale, epoch = fit_one(
                name, sequence, label, train_mask, validation_mask, args.seed, args, device)
            probability = predict(model, sequence, test_index, horizon, mean, scale, device)
            fold[name] = dict(metrics(label[test_index], probability), best_epoch=epoch,
                              rung=RUNG[name])
            fold.setdefault("_correct", {})[name] = (
                (probability >= 0.5).astype(np.float64) == label[test_index]).astype(np.float64)
        folds.append(fold)
        print(f"fold test={fold['test_family']:14s} " +
              "  ".join(f"{n.split('_')[0][:6]}={fold[n]['auroc']:.4f}" for n in METHODS))

    # Pool the per-episode correctness across folds for the paired comparison.
    comparisons = {}
    for name in METHODS:
        if name == TOP:
            continue
        top = np.concatenate([f["_correct"][TOP] for f in folds])
        low = np.concatenate([f["_correct"][name] for f in folds])
        difference, (lo, hi) = episode_bootstrap(top, low, seed=20260903 + int(RUNG[name] * 10))
        comparisons[name] = {
            "rung": RUNG[name], "accuracy_difference": difference,
            "episode_bootstrap_95": [float(lo), float(hi)],
            "indistinguishable_from_rung4": bool(lo <= 0.0 <= hi),
        }
    for f in folds:
        f.pop("_correct")

    report = {
        "schema_version": 1, "benchmark": "ALOHA-provenance",
        "protocol": "leave-one-task-out provenance classification",
        "claim_boundary": audit["claim_boundary"],
        "seed": args.seed, "horizon": max(args.horizons),
        "aggregate": {n: float(np.mean([f[n]["auroc"] for f in folds])) for n in METHODS},
        "comparisons_to_rung4": comparisons,
        "folds": folds,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\nmean macro-AUROC by rung:")
    for name in sorted(METHODS, key=lambda n: RUNG[n]):
        print(f"  R{RUNG[name]:<4} {name:22s} {report['aggregate'][name]:.4f}")
    print("\npaired accuracy difference vs rung 4:")
    for name, c in comparisons.items():
        lo, hi = c["episode_bootstrap_95"]
        flag = "  MATCH" if c["indistinguishable_from_rung4"] else ""
        print(f"  {name:22s} {c['accuracy_difference']:+.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
