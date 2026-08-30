#!/usr/bin/env python3
"""Run frozen causal-head and sensor-space variants on the selected policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_task(spec: dict, selection: dict, task_index: int) -> dict:
    selected = selection.get("selected")
    if not selected or selection.get("all_candidates_ineligible"):
        raise ValueError("integrated selection produced no eligible policy")
    candidates = [
        item for item in selection.get("candidates", [])
        if item.get("label") == selected and item.get("eligible") is True
    ]
    if len(candidates) != 1:
        raise ValueError("selected policy lacks one eligible selection record")
    try:
        policy_config_path = spec["policy_configs"][selected]
    except KeyError as error:
        raise ValueError(f"selected policy has no frozen config mapping: {selected}") from error
    policy = load(policy_config_path)
    if len(policy.get("experiments", [])) != 1:
        raise ValueError("selected ablation requires exactly one policy experiment")
    seeds = [int(seed) for seed in policy["seeds"]]
    variants = spec["variants"]
    task_count = len(variants) * len(seeds)
    if not 0 <= task_index < task_count:
        raise ValueError(f"task-index must be in [0, {task_count - 1}]")
    variant_index, seed_index = divmod(task_index, len(seeds))
    return {
        "selected": selected,
        "policy_config": policy_config_path,
        "policy": policy,
        "variant": variants[variant_index],
        "seed": seeds[seed_index],
        "seed_index": seed_index,
        "task_count": task_count,
    }


def evaluation_filename(
    condition: str, progress: str, visual: str, environment: str = "nominal",
) -> str:
    suffix = []
    if condition != "configured":
        suffix.append(condition)
    if progress != "normal":
        suffix.append(f"progress_{progress}")
    if visual != "none":
        suffix.append(f"visual_{visual}")
    if environment != "nominal":
        suffix.append(f"env_{environment}")
    return "heldout_eval" + ("_" + "_".join(suffix) if suffix else "") + ".json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
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
    run_dir = (
        Path(args.output) / policy["name"] / experiment["method"] / f"seed_{seed}"
    )
    completion_path = run_dir / "TRAINING_COMPLETE.json"
    if not completion_path.exists():
        raise FileNotFoundError(f"selected training is incomplete: {completion_path}")
    variant = resolved["variant"]
    outputs = []
    evaluator = Path(__file__).with_name("evaluate_visual_recovery_ppo.py")
    for condition in spec["conditions"]:
        command = [
            sys.executable, str(evaluator),
            "--config", resolved["policy_config"],
            "--output", args.output,
            "--task-index", str(resolved["seed_index"]),
            "--episodes", str(spec["episodes"]),
            "--num-envs", str(spec["num_envs"]),
            "--seed-base", str(spec["seed_base"]),
            "--condition", condition,
            "--progress-head-mode", variant["progress_head_mode"],
            "--visual-perturbation", variant["visual_perturbation"],
            "--environment-profile", variant.get("environment_profile", "nominal"),
        ]
        subprocess.run(command, check=True)
        output = run_dir / evaluation_filename(
            condition, variant["progress_head_mode"],
            variant["visual_perturbation"],
            variant.get("environment_profile", "nominal"),
        )
        if not output.exists():
            raise FileNotFoundError(f"evaluator did not write expected output: {output}")
        outputs.append({"condition": condition, "path": str(output), "sha256": sha256(output)})
    payload = {
        "schema_version": 1,
        "protocol": "selected-policy causal-head and sensor-space evaluation execution",
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "selected": resolved["selected"],
        "policy_config": resolved["policy_config"],
        "policy_config_sha256": sha256(resolved["policy_config"]),
        "training_seed": seed,
        "variant": variant,
        "outputs": outputs,
        "source_sha256": {
            "spec": sha256(spec_path), "runner": sha256(Path(__file__)),
            "evaluator": sha256(evaluator),
        },
        "claim_boundary": spec["claim_boundary"],
    }
    atomic_json(run_dir / f"ablation_execution_{variant['name']}.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
