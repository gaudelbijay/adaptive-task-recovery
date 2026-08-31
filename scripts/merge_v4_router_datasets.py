#!/usr/bin/env python3
"""Merge router datasets while preserving disjoint simulator split groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--metadata", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--reuse-existing-output", action="store_true")
    args = parser.parse_args()
    if len(args.input) != len(args.metadata):
        raise ValueError("each input requires one metadata file")

    arrays = [np.load(path) for path in args.input]
    metadata = [json.loads(path.read_text()) for path in args.metadata]
    keys = tuple(arrays[0].files)
    if any(tuple(array.files) != keys for array in arrays[1:]):
        raise ValueError("dataset schemas differ")
    contract = metadata[0]["feature_names"]
    if any(item["feature_names"] != contract for item in metadata[1:]):
        raise ValueError("feature contracts differ")
    if any(item.get("prefix_timestamp") != "pre_action_observation_matching_deployment" for item in metadata):
        raise ValueError("input dataset has an unaligned prefix timestamp")
    if any(item.get("hand_engineered_temporal_features") is not False for item in metadata):
        raise ValueError("input dataset contains unaudited temporal feature summaries")

    total_rows = sum(int(array["length"].shape[0]) for array in arrays)
    total_groups = sum(int(item["simulator_batch_groups"]) for item in metadata)
    # Write one key and one source shard at a time. A direct np.concatenate of
    # padded 96-step sequences can transiently require tens of GB.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_existing_output:
        with tempfile.TemporaryDirectory(prefix="router-merge-", dir=args.output.parent) as temporary:
            temporary_path = Path(temporary)
            for key in keys:
                sample = arrays[0][key]
                shape = (total_rows, *sample.shape[1:])
                dtype = sample.dtype
                del sample
                target = np.lib.format.open_memmap(
                    temporary_path / f"{key}.npy", mode="w+", dtype=dtype, shape=shape,
                )
                row_offset = 0
                group_offset = 0
                for array in arrays:
                    source = array[key]
                    count = len(source)
                    if key == "group_id":
                        _, inverse = np.unique(source, return_inverse=True)
                        target[row_offset:row_offset + count] = inverse.astype(np.int64) + group_offset
                        group_offset += int(inverse.max()) + 1
                    else:
                        target[row_offset:row_offset + count] = source
                    row_offset += count
                    del source
                target.flush(); del target
                if key == "group_id":
                    total_groups = group_offset
            temporary_output = args.output.with_name(f".{args.output.name}.tmp")
            with zipfile.ZipFile(
                temporary_output, mode="w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=4, allowZip64=True,
            ) as archive:
                for key in keys:
                    archive.write(temporary_path / f"{key}.npy", arcname=f"{key}.npy")
            temporary_output.replace(args.output)
    elif not args.output.is_file():
        raise FileNotFoundError("--reuse-existing-output requires the completed archive")
    source_manifest = [
        {
            "data": str(data_path),
            "data_sha256": file_sha256(data_path),
            "metadata": str(metadata_path),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "collection_policy": item.get("collection_policy", "merged source archive"),
            "rows": item["rows"],
        }
        for data_path, metadata_path, item in zip(args.input, args.metadata, metadata)
    ]
    result = {
        "schema_version": 1,
        "conditions": metadata[0]["conditions"],
        "snapshots": metadata[0]["snapshots"],
        "rows": total_rows,
        "simulator_batch_groups": total_groups,
        "feature_names": contract,
        "forbidden_feature_keys": metadata[0]["forbidden_feature_keys"],
        "split_unit": "entire vectorized simulator reset batch",
        "prefix_timestamp": "pre_action_observation_matching_deployment",
        "absolute_pose_features": False,
        "hand_engineered_temporal_features": False,
        "training_only_targets": metadata[0]["training_only_targets"],
        "sources": source_manifest,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in {"feature_names", "sources"}}, indent=2))


if __name__ == "__main__":
    main()
