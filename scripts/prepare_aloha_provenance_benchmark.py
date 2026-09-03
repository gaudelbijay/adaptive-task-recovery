#!/usr/bin/env python3
"""Build a second external held-out-family benchmark from ALOHA demonstrations.

`REBOOT` gave the audit one externally collected benchmark. One is a narrow base
for a claim about how held-out-family protocols behave, so this adds a second,
built by a different group, with the same structure: a binary label that is
shared across families, and families that can be held out one at a time.

The label is demonstration *provenance* -- teleoperated by a human, or produced
by a scripted policy. It is annotated rather than inferred: the two conditions
are collected and released as separate datasets. The held-out axis is the task,
so a model must identify provenance on a task it never trained on. Whether that
requires temporal structure is exactly the kind of claim the ladder audits, and
it matters in practice, since provenance detection is used to curate mixed
demonstration corpora.

Only the proprioception and action columns are read. Parquet is columnar and the
Hub filesystem performs HTTP range requests, so the embedded video is never
transferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

# Two tasks, each collected under both provenances. Family = task, label =
# provenance, mirroring REBOOT's object/recovery split.
REPOSITORIES = (
    ("lerobot/aloha_sim_insertion_human",        "insertion",    0),
    ("lerobot/aloha_sim_insertion_scripted",     "insertion",    1),
    ("lerobot/aloha_sim_transfer_cube_human",    "transfer_cube", 0),
    ("lerobot/aloha_sim_transfer_cube_scripted", "transfer_cube", 1),
)


def episode_features(path: str, minimum_frames: int, filesystem: HfFileSystem):
    table = pq.read_table(path, columns=["observation.state", "action", "episode_index"],
                          filesystem=filesystem)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    episode = np.asarray(table["episode_index"].to_pylist()).reshape(-1)
    if state.ndim != 2 or state.shape[1] != 14 or action.shape != state.shape:
        raise RuntimeError(f"unexpected schema in {path}: {state.shape}, {action.shape}")

    out = []
    for index in np.unique(episode):
        rows = episode == index
        s, a = state[rows], action[rows]
        if len(s) < minimum_frames:
            continue
        s, a = s[:minimum_frames], a[:minimum_frames]
        ds = np.diff(s, axis=0, prepend=s[:1])
        da = np.diff(a, axis=0, prepend=a[:1])
        # Relative coordinates, as in the REBOOT benchmark: an absolute home pose
        # would otherwise let a model identify the task rather than the label.
        # No episode duration and no repository identifier is exposed.
        out.append((int(index), np.concatenate(
            (s - s[:1], a - s[:1], a - s, ds + 0.5 * da), axis=1).astype(np.float32)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/aloha/aloha_provenance_v1.npz")
    parser.add_argument("--audit-output", default="results/aloha/aloha_provenance_v1.audit.json")
    parser.add_argument("--frames", type=int, default=256,
                        help="Fixed prefix length; episodes shorter than this are dropped.")
    args = parser.parse_args()

    filesystem = HfFileSystem()
    sequences, labels, families, episodes, sources = [], [], [], [], []
    family_index: dict[str, int] = {}

    for repo, family, label in REPOSITORIES:
        family_index.setdefault(family, len(family_index))
        listing = filesystem.glob(f"datasets/{repo}/data/**/*.parquet")
        if not listing:
            raise SystemExit(f"no parquet found for {repo}")
        kept = 0
        for path in sorted(listing):
            for index, features in episode_features(path, args.frames, filesystem):
                sequences.append(features)
                labels.append(label)
                families.append(family_index[family])
                episodes.append(index)
                sources.append(repo)
                kept += 1
        print(f"{repo:44s} family={family:14s} label={label}  episodes kept={kept}")

    packed = {
        "sequence": np.stack(sequences),
        "label": np.asarray(labels, dtype=np.int64),
        "object_id": np.asarray(families, dtype=np.int64),  # name kept for the shared loader
        "episode_id": np.asarray(episodes, dtype=np.int64),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)

    audit = {
        "schema_version": 1,
        "benchmark": "ALOHA-provenance",
        "protocol": "leave-one-task-out provenance classification from a fixed prefix",
        "claim_boundary": (
            "Offline classification of demonstration provenance on externally "
            "collected data. Not a control result and not a recovery benchmark."
        ),
        "label_meaning": {"0": "human teleoperation", "1": "scripted policy"},
        "families": {name: index for name, index in family_index.items()},
        "prefix_frames": args.frames,
        "episodes": int(len(labels)),
        "human_episodes": int((packed["label"] == 0).sum()),
        "scripted_episodes": int((packed["label"] == 1).sum()),
        "feature_dim": int(packed["sequence"].shape[-1]),
        "repositories": [r for r, _, _ in REPOSITORIES],
    }
    Path(args.audit_output).write_text(json.dumps(audit, indent=2) + "\n")
    print(f"\n{audit['episodes']} episodes, {audit['feature_dim']}-dim features, "
          f"{len(family_index)} families")
    print(f"wrote {output} and {args.audit_output}")


if __name__ == "__main__":
    main()
