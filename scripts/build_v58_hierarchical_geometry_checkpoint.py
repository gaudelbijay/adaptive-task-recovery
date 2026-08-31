#!/usr/bin/env python3
"""Compose frozen V54 tensors with V39's frozen magnitude detector."""

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
    v54_path = Path(task["source_v54_checkpoint"].format(seed=seed))
    v39_path = Path(task["source_v39_checkpoint"].format(seed=seed))
    v54 = torch.load(v54_path, map_location="cpu", weights_only=False)
    v39 = torch.load(v39_path, map_location="cpu", weights_only=False)
    if v54.get("training_protocol") != "continuous_geometry_composition_v19":
        raise ValueError("V58 requires V54")
    if v39.get("training_protocol") != "backkey_targeted_dense_repair_v19":
        raise ValueError("V58 requires V39")
    if v54.get("observation_contract") != v39.get("observation_contract"):
        raise ValueError("V54/V39 observation contracts differ")
    checkpoint = copy.deepcopy(v54)
    checkpoint["agent"] = {
        **v54["agent"],
        **{f"detector.{key}": value for key, value in v39["agent"].items()},
    }
    checkpoint["task"] = task
    checkpoint.pop("optimizer", None)
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V58 run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    completion = json.loads((v54_path.parent / "TRAINING_COMPLETE.json").read_text())
    completion.update({
        "source_v54_checkpoint": str(v54_path),
        "source_v54_checkpoint_sha256": file_sha256(v54_path),
        "source_v39_detector_checkpoint": str(v39_path),
        "source_v39_detector_checkpoint_sha256": file_sha256(v39_path),
        "source_sha256": {
            **completion.get("source_sha256", {}),
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
