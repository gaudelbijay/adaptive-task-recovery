#!/usr/bin/env python3
"""Build an auditable actor-input, supervision, and interaction-cost table."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visual_actor_inputs(task):
    inputs = ["RGB", "robot qpos/qvel", "instruction"]
    if task.get("actor_tcp_pose", False):
        inputs.append("robot TCP pose")
    if task.get("actor_goal_progress", False):
        inputs.append("environment goal progress")
    if task.get("actor_learned_goal_progress", False):
        inputs.append("RGB-predicted goal progress")
    return " + ".join(inputs)


def method_record(spec):
    path = Path(spec["config"])
    config = json.loads(path.read_text(encoding="utf-8"))
    index = int(spec.get("experiment_index", 0))
    if not 0 <= index < len(config["experiments"]):
        raise ValueError(f"invalid experiment index for {path}")
    task = config["experiments"][index]
    modality = spec["modality"]
    if modality not in {"visual", "state"}:
        raise ValueError(f"unknown modality: {modality}")
    batch = int(task["num_envs"]) * int(task["num_steps"])
    requested = int(task["total_timesteps"])
    executed = requested // batch * batch
    dagger = int(task.get("bc_pretrain_updates", 0)) * int(task["num_envs"])
    pose_labels = bool(
        float(task.get("privileged_aux_coefficient", 0.0)) > 0
    )
    progress_labels = bool(task.get("actor_learned_goal_progress", False))
    return {
        "label": spec["label"],
        "method": task.get("method", task["env_id"]),
        "config": str(path), "config_sha256": sha256(path),
        "modality": modality,
        "deployed_actor_inputs": (
            visual_actor_inputs(task) if modality == "visual"
            else "privileged flattened simulator state"
        ),
        "training_only_asymmetric_critic": bool(task.get("asymmetric_critic", False)),
        "training_only_state_teacher": bool(
            task.get("bc_teacher_checkpoint")
            or task.get("bc_nominal_visual_teacher_checkpoint")
            or task.get("bc_strict_state_teacher_checkpoint")
        ),
        "training_only_pose_labels": pose_labels,
        "training_only_goal_resolution_labels": progress_labels,
        "temporal_ssl_coefficient": float(task.get("temporal_ssl_coefficient", 0.0)),
        "vicreg_variance_coefficient": float(task.get("temporal_variance_coefficient", 0.0)),
        "vicreg_covariance_coefficient": float(task.get("temporal_covariance_coefficient", 0.0)),
        "training_seeds": [int(seed) for seed in config["seeds"]],
        "requested_ppo_interactions_per_seed": requested,
        "executed_ppo_interactions_per_seed": executed,
        "dagger_interactions_per_seed": dagger,
        "new_interactions_per_seed": executed + dagger,
        "new_interactions_all_seeds": (executed + dagger) * len(config["seeds"]),
        "initializer_checkpoint": task.get("init_checkpoint"),
        "teacher_checkpoint": task.get("bc_teacher_checkpoint"),
        "nominal_visual_teacher_checkpoint": task.get(
            "bc_nominal_visual_teacher_checkpoint"
        ),
        "strict_state_teacher_checkpoint": task.get(
            "bc_strict_state_teacher_checkpoint"
        ),
        "initializer_label": spec.get("initializer_label"),
        "teacher_label": spec.get("teacher_label"),
        "reported_interactions_exclude_upstream_training": bool(
            task.get("init_checkpoint") or task.get("bc_teacher_checkpoint")
            or task.get("bc_nominal_visual_teacher_checkpoint")
            or task.get("bc_strict_state_teacher_checkpoint")
        ),
        "training_intervention_probability": task.get("env_kwargs", {}).get(
            "intervention_probability"
        ),
        "selection_intervention_probability": task.get("eval_env_kwargs", {}).get(
            "intervention_probability"
        ),
        "learning_rate": float(config["learning_rate"]),
        "claim_boundary": config.get("claim_boundary"),
    }


def build(manifest):
    records = [method_record(spec) for spec in manifest["methods"]]
    labels = [record["label"] for record in records]
    methods = [record["method"] for record in records]
    if len(labels) != len(set(labels)) or len(methods) != len(set(methods)):
        raise ValueError("method-information contract contains duplicate entries")
    return {
        "schema_version": 1,
        "protocol": "configuration-derived method information and interaction accounting",
        "methods": records,
        "claim_boundary": manifest["claim_boundary"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def markdown(payload):
    lines = [
        "# Method information and interaction contract", "",
        "| Method | Actor inputs | Privileged training | Init / teacher lineage | Temporal / VICReg | Seeds | New PPO / seed | New DAgger / seed |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for item in payload["methods"]:
        privilege = []
        if item["training_only_asymmetric_critic"]: privilege.append("critic")
        if item["training_only_state_teacher"]: privilege.append("teacher")
        if item["training_only_pose_labels"]: privilege.append("pose labels")
        if item["training_only_goal_resolution_labels"]: privilege.append("progress labels")
        ssl = f"{item['temporal_ssl_coefficient']:g} / {item['vicreg_variance_coefficient']:g},{item['vicreg_covariance_coefficient']:g}"
        lineage = " / ".join(
            value for value in (item["initializer_label"], item["teacher_label"])
            if value
        ) or "none"
        lines.append(
            f"| {item['label']} | {item['deployed_actor_inputs']} | "
            f"{', '.join(privilege) or 'none'} | {lineage} | {ssl} | "
            f"{len(item['training_seeds'])} | {item['executed_ppo_interactions_per_seed']:,} | "
            f"{item['dagger_interactions_per_seed']:,} |"
        )
    lines.extend([
        "", "Interaction columns count newly collected interactions for the listed "
        "stage; upstream initializer/teacher training is disclosed in the lineage "
        "column and exact checkpoint paths are retained in JSON/CSV, not silently "
        "folded into an unverifiable total.",
        "", f"Claim boundary: {payload['claim_boundary']}.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    payload = build(json.loads(config_bytes))
    payload["config"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        prefix.with_suffix(".json"): json.dumps(payload, indent=2, sort_keys=True) + "\n",
        prefix.with_suffix(".md"): markdown(payload),
    }
    for path, content in outputs.items():
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    csv_path = prefix.with_suffix(".csv")
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        fields = list(payload["methods"][0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in payload["methods"]:
            writer.writerow({
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            })
    os.replace(temporary, csv_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
