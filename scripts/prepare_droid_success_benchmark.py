#!/usr/bin/env python3
"""Build a second external held-out-family benchmark from DROID.

The audit's headline claim -- that a pooled verdict is optimistic because it
averages over families that disagree -- rested on one external benchmark. One is
a narrow base, and an earlier attempt to add a second failed because it had only
two families.

DROID has the structure the ladder needs, verified before this script was
written by `check_droid_family_structure.py`: `is_episode_successful` is an
annotated per-episode label, `building` records where each episode was collected
across the thirteen contributing institutions, and 23 buildings carry both
outcome classes with at least twenty episodes each.

The task is to predict, from a fixed prefix of proprioception and actions,
whether an episode ends in success, at a building the model never trained on.
That is a real question: success detection is used to filter demonstration
corpora and to fit reward models, and it is usually assumed to need temporal
context. Only state and action columns are read; parquet is columnar and the Hub
filesystem issues range requests, so no video is transferred.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "cadene/droid_1.0.1"
COLUMNS = ["observation.state", "action", "building", "is_episode_successful"]


def episode(path: str, frames: int, filesystem: HfFileSystem):
    """Return (building, label, features) for one episode, or None if too short."""
    try:
        table = pq.read_table(path, columns=COLUMNS, filesystem=filesystem)
    except Exception:
        return None
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or len(state) < frames or action.shape[0] != state.shape[0]:
        return None
    building = str(table["building"][0].as_py())
    label = bool(table["is_episode_successful"][0].as_py())

    s, a = state[:frames], action[:frames, : state.shape[1]]
    ds = np.diff(s, axis=0, prepend=s[:1])
    da = np.diff(a, axis=0, prepend=a[:1])
    # Relative coordinates, as on the other external benchmark: an absolute home
    # pose would let a model identify the building rather than the outcome.
    # Episode duration is not exposed, since it correlates with success.
    features = np.concatenate((s - s[:1], a - s[:1], a - s, ds + 0.5 * da),
                              axis=1).astype(np.float32)
    return building, label, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/droid/droid_success_v1.npz")
    parser.add_argument("--audit-output", default="results/droid/droid_success_v1.audit.json")
    parser.add_argument("--shards", type=int, default=12000,
                        help="Episodes to read; DROID stores one per parquet shard.")
    parser.add_argument("--frames", type=int, default=128,
                        help="Fixed prefix length at 15 fps; shorter episodes are dropped.")
    parser.add_argument("--families", type=int, default=10,
                        help="Keep this many largest buildings that carry both classes.")
    parser.add_argument("--min-per-class", type=int, default=15)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{REPO}/data/**/*.parquet"))
    step = max(1, len(shards) // args.shards)
    sample = shards[::step][:args.shards]
    print(f"{len(shards)} shards available; reading {len(sample)}")

    collected = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(lambda p: episode(p, args.frames, filesystem), sample)):
            if result is not None:
                collected.append(result)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(sample)} read, {len(collected)} usable")
    print(f"episodes with >= {args.frames} frames: {len(collected)}")

    # Keep only buildings carrying both outcomes: a held-out family with one
    # class has an undefined AUROC and cannot be scored.
    counts = defaultdict(lambda: [0, 0])
    for building, label, _ in collected:
        counts[building][int(label)] += 1
    eligible = [b for b, (bad, good) in counts.items()
                if bad >= args.min_per_class and good >= args.min_per_class]
    eligible.sort(key=lambda b: -sum(counts[b]))
    keep = eligible[:args.families]
    if len(keep) < 5:
        raise SystemExit(f"only {len(keep)} usable families; the ladder needs at least five")
    index = {b: i for i, b in enumerate(keep)}

    sequences, labels, families = [], [], []
    for building, label, features in collected:
        if building in index:
            sequences.append(features)
            labels.append(int(label))
            families.append(index[building])

    packed = {
        "sequence": np.stack(sequences),
        "label": np.asarray(labels, dtype=np.int64),
        "object_id": np.asarray(families, dtype=np.int64),  # name kept for the shared loader
        "episode_id": np.arange(len(labels), dtype=np.int64),
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **packed)

    audit = {
        "schema_version": 1,
        "benchmark": "DROID-success",
        "protocol": "leave-one-building-out episode-success prediction from a fixed prefix",
        "claim_boundary": (
            "Offline prediction of episode outcome on externally collected real-robot "
            "data. Not a control result, and not a recovery benchmark."
        ),
        "label_meaning": {"0": "episode not successful", "1": "episode successful"},
        "families": index,
        "prefix_frames": args.frames,
        "episodes": int(len(labels)),
        "failures": int((packed["label"] == 0).sum()),
        "successes": int((packed["label"] == 1).sum()),
        "feature_dim": int(packed["sequence"].shape[-1]),
        "per_family": {b: {"failure": counts[b][0], "success": counts[b][1]} for b in keep},
        "repository": REPO,
    }
    Path(args.audit_output).write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\n{audit['episodes']} episodes, {audit['feature_dim']}-dim, {len(keep)} families")
    print(f"  {audit['failures']} failures / {audit['successes']} successes")
    for b in keep:
        print(f"    {b[:34]:<36} fail={counts[b][0]:>4} succ={counts[b][1]:>4}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
