#!/usr/bin/env python3
"""Select full-task visual checkpoints using training-time metrics only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


def metric(metrics, names, default=0.0):
    for name in names:
        if name in metrics:
            return float(metrics[name])
    return float(default)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_selection_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo/visual_recovery_selection_v1")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    summary = {"schema_version": 1, "selection_data": "training-time checkpoint metrics", "seeds": {}}
    for seed in config["seeds"]:
        candidates = []
        for candidate in config["candidates"]:
            path = Path(candidate["checkpoint"].format(seed=seed))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            metrics = checkpoint.get("best_metrics", {})
            success = metric(metrics, ("success_once", "success_at_end", "success"))
            failure = metric(metrics, ("constraint_violated", "fail_once", "fail_at_end"))
            episode_return = metric(metrics, ("return",), default=float("-inf"))
            score = success - float(config["failure_penalty"]) * failure + 1e-6 * episode_return
            candidates.append({
                "name": candidate["name"], "checkpoint": str(path), "score": score,
                "success": success, "failure": failure, "return": episode_return,
                "checkpoint_global_step": int(checkpoint["global_step"]),
                "observation_contract": checkpoint.get("observation_contract"),
            })
        contracts = {candidate["observation_contract"] for candidate in candidates}
        if contracts != {"rgb_qpos_qvel_tcp_instruction_v2"}:
            raise ValueError(f"seed {seed} candidates have incompatible contracts: {contracts}")
        selected = max(candidates, key=lambda item: (item["score"], item["success"], item["return"]))
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        destination = seed_dir / "selected.pt"
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        shutil.copy2(selected["checkpoint"], temporary)
        os.replace(temporary, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        record = {"selected": selected, "candidates": candidates, "selected_sha256": digest}
        (seed_dir / "selection.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        summary["seeds"][str(seed)] = record
    (output / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
