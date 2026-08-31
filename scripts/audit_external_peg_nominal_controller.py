#!/usr/bin/env python3
"""Fail-fast competence audit for ManiSkill's official PegInsertion solver."""

from __future__ import annotations

import argparse
import json
import time

import gymnasium as gym

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401
from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve


def scalar_bool(value) -> bool:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=421_000_100)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--minimum-success-rate", type=float, default=0.75)
    args = parser.parse_args()

    env = gym.make(
        "PegInsertionRecovery-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        intervention_probability=0.0,
        intervention_types=("positive_lateral_peg_ejection",),
    )
    records = []
    started = time.monotonic()
    try:
        for offset in range(args.seed_offset, args.seed_offset + args.episodes):
            seed = args.seed_base + offset
            try:
                transition = solve(env, seed=seed, debug=False, vis=False)
                planning_failed = transition == -1
                info = {} if planning_failed else transition[-1]
                success = False if planning_failed else scalar_bool(info["success"])
                violation = (
                    False if planning_failed
                    else scalar_bool(info["constraint_violated"])
                )
                error = None
            except Exception as exc:  # retain every fail-fast failure in the audit
                planning_failed = True
                success = False
                violation = False
                error = f"{type(exc).__name__}: {exc}"
            records.append({
                "seed": seed,
                "native_success": success,
                "constraint_violation": violation,
                "safe_success": success and not violation,
                "planning_failed": planning_failed,
                "error": error,
            })
    finally:
        env.close()

    safe_successes = sum(row["safe_success"] for row in records)
    result = {
        "schema_version": 1,
        "audit": "official_nominal_controller_competence",
        "environment": "PegInsertionRecovery-v1",
        "base_benchmark": "PegInsertionSide-v1",
        "controller": "ManiSkill3 official Panda motion-planning solution",
        "intervention_probability": 0.0,
        "episodes": args.episodes,
        "seed_offset": args.seed_offset,
        "safe_successes": safe_successes,
        "safe_success_rate": safe_successes / args.episodes,
        "native_success_rate": sum(row["native_success"] for row in records)
        / args.episodes,
        "constraint_violation_rate": sum(
            row["constraint_violation"] for row in records
        ) / args.episodes,
        "planning_failure_rate": sum(row["planning_failed"] for row in records)
        / args.episodes,
        "minimum_success_rate": args.minimum_success_rate,
        "pass": safe_successes / args.episodes >= args.minimum_success_rate,
        "wall_time_seconds": time.monotonic() - started,
        "episodes_detail": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
