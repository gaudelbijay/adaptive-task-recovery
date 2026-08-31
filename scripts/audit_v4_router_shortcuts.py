#!/usr/bin/env python3
"""Audit future-mechanism and hand-engineered-history shortcuts in router data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def group_bucket(groups: np.ndarray) -> np.ndarray:
    return np.asarray([
        int(hashlib.sha256(str(int(value)).encode()).hexdigest()[:8], 16) % 100
        for value in groups
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-length", type=int, default=8)
    args = parser.parse_args()
    raw = np.load(args.data)
    metadata = json.loads(args.metadata.read_text())
    if metadata.get("hand_engineered_temporal_features") is not False:
        raise RuntimeError("feature contract does not exclude temporal summaries")
    forbidden = set(metadata["forbidden_feature_keys"])
    if forbidden & set(metadata["feature_names"]):
        raise RuntimeError("forbidden evaluator labels occur in the feature contract")

    pre_event = (raw["length"] == args.probe_length) & (raw["onset"] >= 12)
    indices = np.flatnonzero(pre_event)
    bucket = group_bucket(raw["group_id"][indices])
    train = bucket < 80
    test = bucket >= 80
    x = raw["sequence"][indices, args.probe_length - 1]
    y = raw["condition"][indices]
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260831),
    )
    probe.fit(x[train], y[train])
    prediction = probe.predict(x[test])
    score = float(balanced_accuracy_score(y[test], prediction))
    result = {
        "schema_version": 1,
        "probe": "future mechanism from instantaneous pre-event observation",
        "probe_length": args.probe_length,
        "classes": int(len(np.unique(y))),
        "chance_balanced_accuracy": float(1 / len(np.unique(y))),
        "balanced_accuracy": score,
        "train_rows": int(train.sum()),
        "test_rows": int(test.sum()),
        "split_unit": "simulator batch group",
        "pass_threshold": 0.30,
        "pass": score <= 0.30,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit("future-mechanism shortcut probe failed")


if __name__ == "__main__":
    main()
