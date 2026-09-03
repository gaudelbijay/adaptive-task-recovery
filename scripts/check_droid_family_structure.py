#!/usr/bin/env python3
"""Check whether DROID has the structure the ladder needs, before building anything.

The audit's per-family analysis needs three things, and an earlier attempt on
ALOHA failed because only the first was checked: at least five families, each
family containing *both* label classes, and enough episodes per family that
holding one out does not collapse training.

DROID annotates `is_episode_successful` and records the `building` each episode
was collected in, so both a label and a family axis exist. Whether they are
distributed usably is an empirical question, and this answers it by reading three
columns. Parquet is columnar and the Hub filesystem does HTTP range requests, so
no video is transferred.
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
COLUMNS = ["episode_index", "building", "is_episode_successful", "task_category"]


def main() -> None:
    parser = argparse.ArgumentParser()
    # DROID stores one episode per shard, so the shard count is the episode
    # count and a usable sample needs to be in the thousands, not the dozens.
    parser.add_argument("--files", type=int, default=2000,
                        help="Parquet shards to sample; DROID stores one episode each.")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--output", default="results/droid/droid_family_structure.json")
    args = parser.parse_args()

    filesystem = HfFileSystem()
    shards = sorted(filesystem.glob(f"datasets/{REPO}/data/**/*.parquet"))
    print(f"shards available: {len(shards)}; sampling {min(args.files, len(shards))}")
    if not shards:
        raise SystemExit("no parquet shards found")
    step = max(1, len(shards) // args.files)
    sample = shards[::step][:args.files]

    # One row per episode: the label and family are constant within an episode.
    def read_one(path):
        try:
            table = pq.read_table(path, columns=COLUMNS, filesystem=filesystem)
        except Exception:
            return []
        idx = np.asarray(table["episode_index"].to_pylist())
        building = np.asarray(table["building"].to_pylist())
        ok = np.asarray(table["is_episode_successful"].to_pylist())
        cat = np.asarray(table["task_category"].to_pylist())
        rows = []
        for e in np.unique(idx):
            first = np.flatnonzero(idx == e)[0]
            rows.append((str(building[first]), bool(ok[first]), str(cat[first])))
        return rows

    episode_label: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, rows in enumerate(pool.map(read_one, sample)):
            for r in rows:
                episode_label[len(episode_label)] = r
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(sample)} shards, {len(episode_label)} episodes")

    by_family = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for building, ok, cat in episode_label.values():
        by_family[building][int(ok)] += 1
        by_cat[cat][int(ok)] += 1

    def report(name, table):
        print(f"\n=== families by {name} ===")
        print(f"{'family':<34}{'fail':>7}{'succ':>7}{'total':>8}{'succ rate':>11}")
        usable = 0
        for key, (bad, good) in sorted(table.items(), key=lambda x: -sum(x[1])):
            total = bad + good
            rate = good / total if total else float("nan")
            both = bad > 0 and good > 0
            usable += both and total >= 20
            flag = "" if both else "   <- single class"
            print(f"{key[:33]:<34}{bad:>7}{good:>7}{total:>8}{rate:>10.3f}{flag}")
        print(f"usable families (both classes, >=20 episodes): {usable}")
        return usable

    usable_building = report("building", by_family)
    usable_category = report("task_category", by_cat)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema_version": 1,
        "repository": REPO,
        "shards_sampled": len(sample),
        "episodes_sampled": len(episode_label),
        "requirement": "at least five families, each with both label classes and >=20 episodes",
        "usable_families_by_building": usable_building,
        "usable_families_by_task_category": usable_category,
        "by_building": {k: {"failure": v[0], "success": v[1]} for k, v in by_family.items()},
        "by_task_category": {k: {"failure": v[0], "success": v[1]} for k, v in by_cat.items()},
    }, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
