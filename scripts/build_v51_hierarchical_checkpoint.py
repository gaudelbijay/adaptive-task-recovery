#!/usr/bin/env python3
"""Repackage V50 weights under hierarchical V47-first routing."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from train_visual_recovery_dual_teacher_ppo import atomic_save, file_sha256, select_task


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument("--task-index", type=int, default=0); parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text()); task, count = select_task(config, args.task_index)
    if args.preflight: print(json.dumps({"task_count": count, **task}, indent=2)); return
    seed = int(task["seed"]); source_path = Path(task["source_visual_checkpoint"].format(seed=seed))
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("training_protocol") != "dedicated_renderer_experts_v19": raise ValueError("V51 requires V50 source")
    run_dir = Path(args.output)/config["name"]/task["method"]/f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()): raise FileExistsError(f"refusing to overwrite V51 run: {run_dir}")
    run_dir.mkdir(parents=True); checkpoint = copy.deepcopy(source)
    checkpoint.update({"training_protocol":"hierarchical_renderer_experts_v19","task":task})
    checkpoint.pop("optimizer", None); atomic_save(checkpoint,run_dir/"best.pt"); atomic_save(checkpoint,run_dir/"latest.pt")
    source_complete=json.loads((source_path.parent/"TRAINING_COMPLETE.json").read_text())
    local=int(source_complete["feature_adapter_transitions"]); initialization=int(source_complete["initialization_simulator_transitions"])
    completion={"schema_version":1,"training_protocol":"hierarchical_renderer_experts_v19","global_step":local,"feature_adapter_transitions":local,"simulator_transitions":local,"ppo_environment_steps":0,"initialization_simulator_transitions":initialization,"total_simulator_transitions":initialization+local,"deployment_actor_inputs":"rgb_qpos_qvel_tcp_instruction_learned_progress","evaluation_domain_label_available":False,"source_visual_checkpoint":str(source_path),"source_visual_checkpoint_sha256":file_sha256(source_path),"source_sha256":{"builder":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (run_dir/"task.json").write_text(json.dumps(task,indent=2)+"\n"); (run_dir/"TRAINING_COMPLETE.json").write_text(json.dumps(completion,indent=2,sort_keys=True)+"\n"); print(json.dumps(completion,indent=2,sort_keys=True))


if __name__ == "__main__": main()
