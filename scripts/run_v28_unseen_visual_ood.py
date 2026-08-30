#!/usr/bin/env python3
"""Run one task from the frozen V28 unseen visual-domain suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from run_selected_visual_causal_ood import (
    evaluation_filename,
    load,
    resolve_task,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(
    evaluator_script: str = "evaluate_visual_recovery_unseen_ood.py",
    execution_protocol: str = "V28 unseen visual-domain execution",
    execution_prefix: str = "unseen_ood_execution",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--task-index", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    spec_path = Path(args.config)
    spec = load(spec_path)
    selection_path = Path(spec["selection"])
    selection = load(selection_path)
    resolved = resolve_task(spec, selection, args.task_index)
    if args.preflight:
        print(json.dumps({
            key: value for key, value in resolved.items() if key != "policy"
        }, indent=2, sort_keys=True))
        return
    policy = resolved["policy"]
    experiment = policy["experiments"][0]
    seed = resolved["seed"]
    run_dir = Path(args.output) / policy["name"] / experiment["method"] / f"seed_{seed}"
    if not (run_dir / "TRAINING_COMPLETE.json").exists():
        raise FileNotFoundError("V28 training is incomplete")
    variant = resolved["variant"]
    evaluator = Path(__file__).with_name(evaluator_script)
    outputs = []
    for condition in spec["conditions"]:
        command = [
            sys.executable, str(evaluator), "--config", resolved["policy_config"],
            "--output", args.output, "--task-index", str(resolved["seed_index"]),
            "--episodes", str(spec["episodes"]), "--num-envs", str(spec["num_envs"]),
            "--seed-base", str(spec["seed_base"]), "--condition", condition,
            "--progress-head-mode", variant["progress_head_mode"],
            "--visual-perturbation", variant["visual_perturbation"],
            "--environment-profile", variant.get("environment_profile", "nominal"),
        ]
        subprocess.run(command, check=True)
        output = run_dir / evaluation_filename(
            condition, variant["progress_head_mode"], variant["visual_perturbation"],
            variant.get("environment_profile", "nominal"),
        )
        outputs.append({"condition": condition, "path": str(output), "sha256": sha256(output)})
    payload = {
        "schema_version": 1,
        "protocol": execution_protocol,
        "selection": str(selection_path), "selection_sha256": sha256(selection_path),
        "selected": resolved["selected"], "policy_config": resolved["policy_config"],
        "policy_config_sha256": sha256(resolved["policy_config"]),
        "training_seed": seed, "variant": variant, "outputs": outputs,
        "source_sha256": {
            "spec": sha256(spec_path), "runner": sha256(Path(__file__)),
            "evaluator": sha256(evaluator),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    atomic_json(run_dir / f"{execution_prefix}_{variant['name']}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
