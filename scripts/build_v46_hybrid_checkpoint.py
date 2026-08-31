#!/usr/bin/env python3
"""Compose the V44 router and V45 encoder without consuming new rollouts."""

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
    task, task_count = select_task(config, args.task_index)
    if args.preflight:
        print(json.dumps({"task_count": task_count, **task}, indent=2))
        return
    seed = int(task["seed"])
    v44_path = Path(task["v44_checkpoint"].format(seed=seed))
    v45_path = Path(task["v45_checkpoint"].format(seed=seed))
    v44 = torch.load(v44_path, map_location="cpu", weights_only=False)
    v45 = torch.load(v45_path, map_location="cpu", weights_only=False)
    expected = "routed_multiview_feature_adapter_v19"
    if v44.get("training_protocol") != expected or v45.get("training_protocol") != expected:
        raise ValueError("V46 source protocol mismatch")
    if v44.get("observation_contract") != v45.get("observation_contract"):
        raise ValueError("V46 source observation contracts differ")
    state = copy.deepcopy(v45["agent"])
    router_keys = [key for key in state if key.startswith("router.")]
    if not router_keys or any(key not in v44["agent"] for key in router_keys):
        raise ValueError("V46 router state is incomplete")
    for key in router_keys:
        state[key] = v44["agent"][key].clone()
    run_dir = Path(args.output) / config["name"] / task["method"] / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite V46 run: {run_dir}")
    run_dir.mkdir(parents=True)
    local_transitions = int(v44["global_step"]) + int(v45["global_step"])
    checkpoint = copy.deepcopy(v45)
    checkpoint.update({
        "training_protocol": "hybrid_calibrated_feature_adapter_v19",
        "task": task,
        "agent": state,
        "global_step": local_transitions,
        "iteration": 0,
    })
    checkpoint.pop("optimizer", None)
    atomic_save(checkpoint, run_dir / "best.pt")
    atomic_save(checkpoint, run_dir / "latest.pt")
    v45_complete = json.loads((v45_path.parent / "TRAINING_COMPLETE.json").read_text())
    initialization = int(v45_complete["initialization_simulator_transitions"])
    completion = {
        "schema_version": 1,
        "training_protocol": "hybrid_calibrated_feature_adapter_v19",
        "global_step": local_transitions,
        "feature_adapter_transitions": local_transitions,
        "simulator_transitions": local_transitions,
        "ppo_environment_steps": 0,
        "initialization_simulator_transitions": initialization,
        "total_simulator_transitions": initialization + local_transitions,
        "deployment_actor_inputs": "rgb_qpos_qvel_tcp_instruction_learned_progress",
        "evaluation_domain_label_available": False,
        "route_threshold": float(task["route_threshold"]),
        "v44_checkpoint": str(v44_path),
        "v44_checkpoint_sha256": file_sha256(v44_path),
        "v45_checkpoint": str(v45_path),
        "v45_checkpoint_sha256": file_sha256(v45_path),
        "source_sha256": {
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
