#!/usr/bin/env python3
"""Test whether the PegInsertion static model is reading `normalized_time`.

Peg prefixes are current-centered, so a model that sees only the final frame
receives near-zero geometry. It nonetheless reaches 0.80 and 0.84 on the two
blockage conditions. The remaining input is the clock, and if that is what it
uses, the benchmark leaks the condition through episode timing rather than
through physics.

This writes a copy of the dataset with `normalized_time` zeroed and nothing
else changed, so the same training command can be run against both and the
accuracies compared directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--feature", default="normalized_time")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    names = metadata["feature_names"]
    if args.feature not in names:
        raise SystemExit(f"{args.feature!r} is not in the feature schema")
    index = names.index(args.feature)

    raw = np.load(args.data)
    arrays = {key: raw[key] for key in raw.files}
    sequence = arrays["sequence"].copy()
    before = float(np.abs(sequence[:, :, index]).mean())
    sequence[:, :, index] = 0.0
    arrays["sequence"] = sequence

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_data, **arrays)

    # The metadata hash is checked at load time, so it must be rewritten to
    # match the new file rather than reused.
    metadata = dict(metadata)
    metadata["ablated_feature"] = args.feature
    metadata["ablation_note"] = (
        f"{args.feature} zeroed in place; all other columns and every label, "
        "split, and group id are unchanged."
    )
    args.output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps({
        "feature": args.feature,
        "feature_index": index,
        "mean_abs_before": before,
        "mean_abs_after": float(np.abs(sequence[:, :, index]).mean()),
        "rows": int(sequence.shape[0]),
        "output_data": str(args.output_data),
        "output_data_sha256": hashlib.sha256(args.output_data.read_bytes()).hexdigest()[:16],
    }, indent=2))


if __name__ == "__main__":
    main()
