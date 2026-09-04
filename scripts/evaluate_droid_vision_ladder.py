#!/usr/bin/env python3
"""Ladder on the DROID vision benchmark, with the visual block reduced per fold.

Handing the rungs 384-dimensional frozen embeddings alongside 32 dimensions of
proprioception collapsed every rung to chance: with roughly 2,400 training
episodes the visual block is 92% of the input width and swamps the signal. That
is a property of how the features were presented, not of the benchmark, so this
reduces the visual block before fitting.

The projection is fitted on the training fold only and applied to validation and
test, so no held-out building influences the basis. Everything else -- the
models, the fitting routine, the metrics, the per-(method, fold) seeding -- is
imported unchanged from the evaluator used for the other benchmarks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_reboot_causal_prefix import fit_one, metrics, predict  # noqa: E402

METHODS = ("static_mlp", "endpoint_pair_mlp", "moment_mlp",
           "unstructured_gru", "causal_dynamics_gru")
RUNG = {"static_mlp": 1, "endpoint_pair_mlp": 2, "moment_mlp": 2.5,
        "unstructured_gru": 4, "causal_dynamics_gru": 4}
TOP = "causal_dynamics_gru"


def reduce_visual(sequence, visual_dim, components, train_mask, seed):
    """PCA the visual block on the training fold; leave proprioception intact.

    Fitting on training rows only matters here: a basis computed over all
    buildings would carry information about the held-out one into the fit.
    """
    visual, proprio = sequence[..., :visual_dim], sequence[..., visual_dim:]
    flat = visual[train_mask].reshape(-1, visual_dim)
    mean = flat.mean(axis=0, keepdims=True)
    centred = flat - mean
    # Randomized SVD on a subsample keeps this cheap; the block is low rank in
    # practice because frames within an episode are highly correlated.
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(centred), size=min(len(centred), 20000), replace=False)
    _, _, vt = np.linalg.svd(centred[rows], full_matrices=False)
    basis = vt[:components].T
    projected = (visual.reshape(-1, visual_dim) - mean) @ basis
    projected = projected.reshape(*visual.shape[:2], components)
    return np.concatenate([projected, proprio], axis=2).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="results/droid/droid_vision_v1.npz")
    parser.add_argument("--audit", default="results/droid/droid_vision_v1.audit.json")
    parser.add_argument("--output", default="results/droid/vision_pca_ladder_seed0.json")
    parser.add_argument("--components", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[8, 16, 32])
    args = parser.parse_args()
    args.train_fraction = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data)
    sequence, label = raw["sequence"], raw["label"]
    family = raw["object_id"]
    audit = json.loads(Path(args.audit).read_text())
    names = audit["objects"]
    visual_dim = audit["visual_dim"]

    folds = []
    for test_family in sorted(np.unique(family)):
        validation_family = (test_family + 1) % len(names)
        train_mask = (family != test_family) & (family != validation_family)
        validation_mask = family == validation_family
        test_mask = family == test_family

        reduced = reduce_visual(sequence, visual_dim, args.components,
                                train_mask, args.seed * 131 + int(test_family))
        fold = {"test_object": names[int(test_family)],
                "validation_object": names[int(validation_family)],
                "components": args.components}
        test_index = np.flatnonzero(test_mask)
        for name in METHODS:
            model, mean, scale, epoch = fit_one(
                name, reduced, label, train_mask, validation_mask, args.seed, args, device)
            probability = predict(model, reduced, test_index, max(args.horizons),
                                  mean, scale, device)
            fold[name] = dict(metrics(label[test_index], probability),
                              best_epoch=epoch, rung=RUNG[name])
        folds.append(fold)
        print(f"fold {fold['test_object'][:22]:<24} " +
              "  ".join(f"{n.split('_')[0][:6]}={fold[n]['auroc']:.3f}" for n in METHODS))

    report = {
        "schema_version": 1, "benchmark": "DROID-success-vision-pca",
        "protocol": "leave-one-building-out, visual block PCA-reduced per fold",
        "claim_boundary": audit["claim_boundary"],
        "components": args.components, "visual_dim": visual_dim,
        "seed": args.seed,
        "aggregate": {n: float(np.mean([f[n]["auroc"] for f in folds])) for n in METHODS},
        "folds": folds,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("\nmean AUROC:")
    for n in sorted(METHODS, key=lambda x: RUNG[x]):
        print(f"  R{RUNG[n]:<4} {n:22s} {report['aggregate'][n]:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
