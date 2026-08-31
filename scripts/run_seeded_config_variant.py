#!/usr/bin/env python3
"""Run one task from a hash-pinned, mechanically derived multi-seed config."""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    args = parser.parse_args()
    manifest = json.loads(Path(args.pipeline_config).read_text())
    stages = {item["id"]: item for item in manifest["stages"]}
    if args.stage not in stages:
        raise KeyError(f"unknown pipeline stage: {args.stage}")
    stage = stages[args.stage]
    base_path = Path(stage["base_config"])
    observed = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if observed != stage["base_sha256"]:
        raise ValueError(f"base config hash mismatch for {args.stage}")
    config = json.loads(base_path.read_text())
    config["name"] = stage["output_name"]
    config["seeds"] = list(manifest["seeds"])
    for experiment in config["experiments"]:
        for key, value in stage.get("experiment_overrides", {}).items():
            if key not in experiment:
                raise KeyError(f"{args.stage} cannot override missing key {key!r}")
            experiment[key] = value
    with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
        json.dump(config, stream, sort_keys=True)
        stream.flush()
        command = [
            sys.executable, stage["entrypoint"], "--config", stream.name,
            "--output", args.output, "--task-index", str(args.task_index),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
