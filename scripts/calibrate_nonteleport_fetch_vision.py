#!/usr/bin/env python3
"""Measure bowl RGB-change after the same physical prefix used at test time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_replicacad_manipulation import attempt_goal_with_real_grasp
from atr.envs.tidy_up_replicacad_policies import _TRAY_SLOTS
from atr.physical_pipeline import instruction_graph, recovery_change_score, settle_before_task


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--intervention", choices=("none", "cracker_box_destroyed"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames-dir")
    parser.add_argument("--settle-steps", type=int, default=0)
    args = parser.parse_args()
    env = gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode="rgb_array",
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=args.intervention, onset_step_range=(5, 6),
    )
    try:
        env.reset(seed=args.seed)
        # Match the live pipeline exactly, including navigation-level guard
        # screening against the parsed physical-task contract.
        env.unwrapped.goal_graph = instruction_graph()
        settle_before_task(env, args.settle_steps)
        protected_before = env.unwrapped._get_actor("master_chef_can").pose.sp.p.copy()
        reference = env.render()[0].cpu().numpy()
        can_result = attempt_goal_with_real_grasp(
            env, instruction_graph().goals[0], _TRAY_SLOTS[0],
        )
        current = env.render()[0].cpu().numpy()
        if args.frames_dir:
            frames_dir = Path(args.frames_dir)
            frames_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reference).save(frames_dir / f"{args.intervention}_seed_{args.seed}_before.png")
            Image.fromarray(current).save(frames_dir / f"{args.intervention}_seed_{args.seed}_after.png")
        payload = {
            "seed": args.seed,
            "intervention": args.intervention,
            "recovery_change_score": recovery_change_score(reference, current),
            "can_result": can_result,
            "oracle_recovery_object_exists": env.unwrapped._exists["cracker_box"],
            "protected_position_before": protected_before,
            "protected_position_after": env.unwrapped._get_actor("master_chef_can").pose.sp.p.copy(),
        }
        payload["protected_displacement"] = float(np.linalg.norm(
            payload["protected_position_after"] - payload["protected_position_before"]
        ))
    finally:
        env.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
