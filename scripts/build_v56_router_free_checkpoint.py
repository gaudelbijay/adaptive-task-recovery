#!/usr/bin/env python3
"""Repackage frozen V54 tensors under the predeclared router-free deployment view."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from train_visual_recovery_dual_teacher_ppo import atomic_save, file_sha256, select_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    task, count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": count, **task}, indent=2))
        return

    seed = int(task["seed"])
    source_path = Path(task["source_v54_checkpoint"].format(seed=seed))
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("training_protocol") != "continuous_geometry_composition_v19":
        raise ValueError("V56 requires a completed V54 source checkpoint")
    source_complete = json.loads((source_path.parent / "TRAINING_COMPLETE.json").read_text())
    if source_complete.get("training_protocol") != source["training_protocol"]:
        raise ValueError("V54 checkpoint/completion protocol mismatch")

    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V56 run: {run_dir}")
    run_dir.mkdir(parents=True)
    checkpoint = copy.deepcopy(source)
    checkpoint["task"] = task
    checkpoint.pop("optimizer", None)
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")

    completion = copy.deepcopy(source_complete)
    completion.update({
        "source_v54_checkpoint": str(source_path),
        "source_v54_checkpoint_sha256": file_sha256(source_path),
        "source_sha256": {
            **source_complete.get("source_sha256", {}),
            "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    })
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
