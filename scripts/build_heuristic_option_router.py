#!/usr/bin/env python3
"""Emit a loadable checkpoint for the hand-written V28 motion-threshold router.

The heuristic baseline has no trained parameters, but the evaluation harness
loads every router the same way and verifies the matched feature metadata
hash.  This script writes a checkpoint carrying that metadata so the baseline
runs through the identical code path as the learned routers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from atr.policies.option_router import OPTION_NAMES
from atr.policies.heuristic_option_router import MOTION_THRESHOLD, HeuristicMotionRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=MOTION_THRESHOLD)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry-dim", type=int, default=57)
    parser.add_argument("--heldout-option", type=int, default=2)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    feature_names = metadata["feature_names"]
    # Constructing the model here validates every feature slice resolves, so a
    # schema change fails at build time rather than mid-evaluation.
    model = HeuristicMotionRouter(feature_names, args.threshold)

    checkpoint = {
        "schema_version": 1,
        "model": "heuristic_motion",
        "seed": args.seed,
        "input_dim": len(feature_names),
        "hidden_dim": 0,
        "threshold": args.threshold,
        "feature_names": feature_names,
        "feature_metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "current_centered_geometry_dim": args.geometry_dim,
        "heldout_option": args.heldout_option,
        "state_dict": model.state_dict(),
        # A deterministic one-hot router is always maximally confident, so the
        # shared abstention path never fires. This is intentional: the V28
        # baseline does not abstain.
        "calibration": {
            "threshold": 0.5,
            "class_thresholds_99_precision": [0.5] * len(OPTION_NAMES),
            "selective_error": None,
            "coverage": 1.0,
            "note": "hand-written baseline; no learned calibration",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": digest,
        "input_dim": len(feature_names),
        "threshold": args.threshold,
        "forward_features": model.forward_index.tolist(),
        "reverse_features": model.reverse_index.tolist(),
        "blocker_features": model.blocker_index.tolist(),
    }, indent=2))


if __name__ == "__main__":
    main()
