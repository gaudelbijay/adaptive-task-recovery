#!/usr/bin/env python3
"""Fail-fast physics audit for the external PegInsertion recovery task."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401


def run_condition(condition: str, num_envs: int, steps: int, seed: int):
    env = gym.make(
        "PegInsertionRecovery-v1",
        num_envs=num_envs,
        reconfiguration_freq=0,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        intervention_probability=0.0 if condition == "nominal" else 1.0,
        intervention_types=(
            ("positive_lateral_peg_ejection",)
            if condition == "nominal" else (condition,)
        ),
        onset_step_range=(2, 2),
        blocker_return_delay_steps=48,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=True, record_metrics=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        initial_y = obs["extra"]["peg_pose"][:, 1].clone()
        maximum_y_shift = torch.zeros(num_envs, device=initial_y.device)
        minimum_hole_distance = torch.full(
            (num_envs,), torch.inf, device=initial_y.device,
        )
        minimum_target_distance = torch.full_like(minimum_hole_distance, torch.inf)
        ever_engaged = torch.zeros(num_envs, dtype=torch.bool, device=initial_y.device)
        ever_cleared = torch.zeros_like(ever_engaged)
        info = None
        action = torch.zeros(
            (num_envs,) + env.single_action_space.shape,
            device=initial_y.device,
        )
        for _ in range(steps):
            obs, _, _, _, info = env.step(action)
            shift = (obs["extra"]["peg_pose"][:, 1] - initial_y).abs()
            maximum_y_shift = torch.maximum(maximum_y_shift, shift)
            distance = torch.linalg.vector_norm(
                obs["extra"]["hole_blocker_pose"][:, :3]
                - obs["extra"]["box_hole_pose"][:, :3],
                dim=1,
            )
            minimum_hole_distance = torch.minimum(minimum_hole_distance, distance)
            minimum_target_distance = torch.minimum(
                minimum_target_distance,
                info["critic_blocker_target_distance"],
            )
            ever_engaged |= info["blocker_engaged"].bool()
            ever_cleared |= info["temporary_cleared"].bool()
        assert info is not None
        result = {
            "condition": condition,
            "episodes": num_envs,
            "finite_observation": bool(torch.isfinite(obs["extra"]["peg_pose"]).all()),
            "maximum_peg_y_shift_mean": float(maximum_y_shift.mean()),
            "ejection_observed_rate": float((maximum_y_shift > 0.01).float().mean()),
            "blocker_engaged_rate": float(ever_engaged.float().mean()),
            "minimum_hole_distance_mean": float(minimum_hole_distance.mean()),
            "minimum_hole_distance_max": float(minimum_hole_distance.max()),
            "minimum_target_distance_mean": float(minimum_target_distance.mean()),
            "minimum_target_distance_max": float(minimum_target_distance.max()),
            "temporary_cleared_rate": float(ever_cleared.float().mean()),
            "constraint_violation_rate": float(
                info["constraint_violated"].float().mean()
            ),
            "native_success_rate_under_zero_action": float(info["success"].float().mean()),
        }
        if not result["finite_observation"]:
            raise RuntimeError(f"non-finite state in {condition}")
        if "ejection" in condition and result["ejection_observed_rate"] < 0.9:
            raise RuntimeError(f"ejection force did not move the peg: {result}")
        if "ejection" in condition and result["constraint_violation_rate"] > 0.1:
            raise RuntimeError(f"ejection is not physically recoverable: {result}")
        if condition in ("permanent_hole_block", "temporary_hole_block"):
            if result["blocker_engaged_rate"] < 0.9:
                raise RuntimeError(f"blocker did not physically engage: {result}")
        if condition == "temporary_hole_block" and result["temporary_cleared_rate"] < 0.9:
            raise RuntimeError(f"temporary blocker did not retract: {result}")
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=421000001)
    args = parser.parse_args()
    conditions = (
        "nominal",
        "positive_lateral_peg_ejection",
        "negative_lateral_peg_ejection",
        "permanent_hole_block",
        "temporary_hole_block",
    )
    results = [
        run_condition(condition, args.num_envs, args.steps, args.seed + index)
        for index, condition in enumerate(conditions)
    ]
    print(json.dumps({"schema_version": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()
