#!/usr/bin/env python3
"""Exit zero iff one immutable manipulation experiment finished its budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tasks = [
        {**experiment, "seed": seed}
        for experiment in config["experiments"]
        for seed in config["seeds"]
    ]
    task = tasks[args.task_index]
    method = task.get("method", task["env_id"])
    checkpoint_path = (
        Path(args.output) / config["name"] / method / f"seed_{task['seed']}" / "latest.pt"
    )
    if not checkpoint_path.exists():
        raise SystemExit(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    batch_size = int(task["num_envs"]) * int(task["num_steps"])
    scheduled_steps = int(task["total_timesteps"]) // batch_size * batch_size
    complete = checkpoint.get("task") == task and int(checkpoint["global_step"]) >= scheduled_steps
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "global_step": int(checkpoint["global_step"]),
        "scheduled_steps": scheduled_steps,
        "complete": complete,
    }))
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
