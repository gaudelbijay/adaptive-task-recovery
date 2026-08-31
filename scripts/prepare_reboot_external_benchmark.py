#!/usr/bin/env python3
"""Download a pinned REBOOT snapshot and build a no-video prefix benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem


def fetch(url: str, path: Path):
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024): output.write(chunk)
    temporary.replace(path)
    return path


def repo_info(entry, cache: Path):
    repo = entry["id"].split("/", 1)[1]
    root = f"https://huggingface.co/datasets/{entry['id']}/resolve/{entry['sha']}"
    info_path = fetch(f"{root}/meta/info.json", cache / repo / "meta_info.json")
    info = json.loads(info_path.read_text())
    if info.get("fps") != 30:
        raise RuntimeError(f"unexpected FPS for {entry['id']}")
    return repo, root, info


def trajectory_features(path: str, minimum_frames: int, filesystem: HfFileSystem):
    # Parquet is columnar.  HfFileSystem performs HTTP range reads, so only
    # proprioception/action chunks and the footer are transferred; embedded
    # RGB-D arrays (hundreds of MB per episode) are never downloaded.
    table = pq.read_table(
        path, columns=["observation.state", "action"], filesystem=filesystem,
    )
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
        raise RuntimeError(f"unexpected trajectory schema in {path}: {state.shape}, {action.shape}")
    if len(state) < minimum_frames:
        return None, len(state)
    state = state[:minimum_frames]
    action = action[:minimum_frames]
    delta_state = np.diff(state, axis=0, prepend=state[:1])
    delta_action = np.diff(action, axis=0, prepend=action[:1])
    # Relative coordinates prevent an object-specific home pose from becoming
    # the benchmark solution.  No episode duration or repository ID is input.
    return np.concatenate((
        state - state[:1], action - state[:1], action - state,
        delta_state + 0.5 * delta_action,
    ), axis=1).astype(np.float32), len(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reboot_external_benchmark_v1.json")
    parser.add_argument("--cache", default="results/reboot_cache_v1")
    parser.add_argument("--output", default="results/reboot/reboot_prefix_v1.npz")
    parser.add_argument("--audit-output", default="results/reboot/reboot_prefix_v1.audit.json")
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    cache = Path(args.cache)
    prefix_frames = max(config["fixed_prefix_frames"])
    objects = sorted({entry["object"] for entry in config["repositories"]})
    object_index = {name: index for index, name in enumerate(objects)}
    trajectory_specs = []
    repository_metadata = []
    for entry in config["repositories"]:
        repo, root, info = repo_info(entry, cache)
        repository_metadata.append({
            "id": entry["id"], "sha": entry["sha"],
            "episodes": int(info["total_episodes"]), "frames": int(info["total_frames"]),
        })
        for episode in range(int(info["total_episodes"])):
            name = f"file-{episode:03d}.parquet"
            trajectory_specs.append((
                f"datasets/{entry['id']}@{entry['sha']}/data/chunk-000/{name}",
                entry, episode,
            ))
    filesystem = HfFileSystem()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        extracted = list(pool.map(
            lambda spec: trajectory_features(spec[0], prefix_frames, filesystem),
            trajectory_specs,
        ))
    sequence, label, object_id, repository, episode_id, hashes = [], [], [], [], [], []
    short_exclusions = []
    for (features, frame_count), (path, entry, episode) in zip(extracted, trajectory_specs):
        if features is None:
            short_exclusions.append({
                "repository": entry["id"], "episode": episode,
                "frames": frame_count, "required_frames": prefix_frames,
            })
            continue
        sequence.append(features)
        label.append(int(entry["recovery"]))
        object_id.append(object_index[entry["object"]])
        repository.append(entry["id"])
        episode_id.append(episode)
        hashes.append(hashlib.sha256(features.tobytes()).hexdigest())
    duplicates = len(hashes) - len(set(hashes))
    packed = {
        "sequence": np.stack(sequence), "label": np.asarray(label, dtype=np.int64),
        "object_id": np.asarray(object_id, dtype=np.int64),
        "repository": np.asarray(repository), "episode_id": np.asarray(episode_id, dtype=np.int64),
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    audit = {
        "schema_version": 1, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset_rows": len(sequence), "nominal_rows": int((packed["label"] == 0).sum()),
        "recovery_rows": int((packed["label"] == 1).sum()), "objects": objects,
        "feature_dim": int(packed["sequence"].shape[-1]), "prefix_frames": prefix_frames,
        "exact_feature_duplicates": duplicates, "repositories": repository_metadata,
        "short_trajectory_exclusions": short_exclusions,
        "anti_shortcut": {
            "fixed_duration": True, "trajectory_length_input": False,
            "repository_id_input": False, "absolute_home_pose_removed": True,
            "split_unit": "object",
        },
        "public_snapshot_note": (
            "The pinned named-task repositories expose fewer episodes than the "
            "2,160 stated by the project page; counts above describe exactly the public files used."
        ),
    }
    Path(args.audit_output).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
