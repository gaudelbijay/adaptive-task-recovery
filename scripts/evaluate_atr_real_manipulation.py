#!/usr/bin/env python3
"""Evaluate the non-teleport Fetch manipulation skill on the full ATR goal list."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_env_replicacad import replicacad_example
from atr.envs.tidy_up_replicacad_manipulation import attempt_goal_with_real_grasp
from atr.envs.tidy_up_replicacad_policies import _TRAY_SLOTS
from atr.feasibility.oracle import evaluate_goal_graph


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", default="results/manipulation_ppo/atr_real_grasp_baseline.json")
    args = parser.parse_args()
    graph = replicacad_example()
    records = []
    for seed in range(args.episodes):
        env = gym.make(
            "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
            sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
            intervention_kind="none",
        )
        try:
            env.reset(seed=seed)
            initial_state = env.unwrapped._world_state()
            outcomes = []
            for goal, tray_slot in zip(graph.goals, _TRAY_SLOTS, strict=True):
                outcomes.append({
                    "goal_id": goal.id,
                    **attempt_goal_with_real_grasp(env, goal, tray_slot),
                })
            final_state = env.unwrapped._world_state()
            oracle = evaluate_goal_graph(graph, initial_state, final_state)
            records.append({
                "seed": seed,
                "goals_achieved": sum(outcome["achieved"] for outcome in outcomes),
                "grasped": sum(outcome["grasped"] for outcome in outcomes),
                "carried": sum(outcome["carried"] for outcome in outcomes),
                "steps_used": sum(outcome["steps_used"] for outcome in outcomes),
                "constraint_violations": sum(oracle["constraint_violations"].values()),
                "outcomes": outcomes,
            })
        finally:
            env.close()
    payload = {
        "schema_version": 1,
        "protocol": "sequential non-teleport Fetch manipulation",
        "episodes": args.episodes,
        "records": records,
        "mean_goals_achieved": sum(row["goals_achieved"] for row in records) / args.episodes,
        "mean_constraint_violations": sum(row["constraint_violations"] for row in records) / args.episodes,
        "complete_task_success_rate": sum(row["goals_achieved"] == len(graph.goals) for row in records) / args.episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
