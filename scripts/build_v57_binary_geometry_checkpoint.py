#!/usr/bin/env python3
"""Compose V55's frozen binary router with V54's trained geometry experts."""

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
    v55_path = Path(task["source_v55_checkpoint"].format(seed=seed))
    v54 = torch.load(v54_path, map_location="cpu", weights_only=False)
    v55 = torch.load(v55_path, map_location="cpu", weights_only=False)
    if v54.get("training_protocol") != "continuous_geometry_composition_v19":
        raise ValueError("V57 requires a V54 geometry checkpoint")
    if v55.get("training_protocol") != "binary_geometry_router_v19":
        raise ValueError("V57 requires a V55 binary-router checkpoint")
    if v54.get("observation_contract") != v55.get("observation_contract"):
        raise ValueError("V54/V55 observation contracts differ")

    v54_state, v55_state = v54["agent"], v55["agent"]
    v54_router_keys = {key for key in v54_state if key.startswith("router.")}
    v55_router_keys = {key for key in v55_state if key.startswith("router.")}
    shared = set(v54_state) - v54_router_keys
    if not v55_router_keys or shared != set(v55_state) - v55_router_keys:
        raise ValueError("V54/V55 state schemas are incompatible")
    trainable_prefixes = ("global_correctors.", "dense_correctors.")
    for key in shared:
        if not key.startswith(trainable_prefixes) and not torch.equal(v54_state[key], v55_state[key]):
            raise ValueError(f"unexpected V54/V55 source-tensor difference: {key}")

    checkpoint = copy.deepcopy(v55)
    checkpoint["agent"] = {
        **{key: v54_state[key] for key in shared},
        **{key: v55_state[key] for key in v55_router_keys},
    }
    checkpoint["training_protocol"] = "continuous_geometry_composition_v19"
    checkpoint["task"] = task
    checkpoint["global_step"] = int(v54["global_step"])
    checkpoint["iteration"] = int(v54["iteration"])
    checkpoint.pop("optimizer", None)

    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V57 run: {run_dir}")
    run_dir.mkdir(parents=True)
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")

    v54_complete = json.loads((v54_path.parent / "TRAINING_COMPLETE.json").read_text())
    v55_complete = json.loads((v55_path.parent / "TRAINING_COMPLETE.json").read_text())
    local = int(v54_complete["geometry_training_transitions"])
    router = int(v55_complete["router_training_transitions"])
    initialization = int(v54_complete["initialization_simulator_transitions"]) + router
    completion = {
        **v54_complete,
        "initialization_simulator_transitions": initialization,
        "total_simulator_transitions": initialization + local,
        "binary_router_training_transitions": router,
        "source_v54_checkpoint": str(v54_path),
        "source_v54_checkpoint_sha256": file_sha256(v54_path),
        "source_v55_checkpoint": str(v55_path),
        "source_v55_checkpoint_sha256": file_sha256(v55_path),
        "source_sha256": {
            **v54_complete.get("source_sha256", {}),
            "builder": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    (run_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n")
    (run_dir / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
