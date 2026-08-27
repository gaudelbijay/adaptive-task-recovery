#!/usr/bin/env python3
"""Run one complete non-teleport Fetch ATR episode and save JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401
from atr.physical_pipeline import run_nonteleport_episode
from atr.policies.q_learning import load_q_table_checkpoint


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy", choices=("static", "oracle", "visual_learned_guarded"),
        default="visual_learned_guarded",
    )
    parser.add_argument("--recovery-change-threshold", type=float, required=True)
    parser.add_argument("--intervention", choices=("cracker_box_destroyed", "none"), default="cracker_box_destroyed")
    args = parser.parse_args()

    env = gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=args.intervention, onset_step_range=(5, 6),
    )
    try:
        env.reset(seed=args.seed)
        result = run_nonteleport_episode(
            env, load_q_table_checkpoint(args.checkpoint),
            recovery_change_threshold=args.recovery_change_threshold, policy=args.policy,
        )
    finally:
        env.close()
    result.update({"seed": args.seed, "policy": args.policy, "intervention": args.intervention})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
