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
from atr.policies.peg_router_features import world_to_local


def run_condition(
    condition: str,
    num_envs: int,
    steps: int,
    seed: int,
    ejection_force: float,
    ejection_steps: int,
    negative_ejection_force_scale: float,
    ejection_target_displacement: float,
    ejection_position_gain: float,
    ejection_velocity_gain: float,
    blocker_force: float,
    blocker_position_gain: float,
    blocker_velocity_gain: float,
    blocker_gravity_compensation: float,
    blocker_home_offset: float,
    blocker_target_peg_length_scale: float,
    blocker_return_position_gain: float,
):
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
        ejection_force=ejection_force,
        ejection_steps=ejection_steps,
        negative_ejection_force_scale=negative_ejection_force_scale,
        ejection_target_displacement=ejection_target_displacement,
        ejection_position_gain=ejection_position_gain,
        ejection_velocity_gain=ejection_velocity_gain,
        blocker_force=blocker_force,
        blocker_position_gain=blocker_position_gain,
        blocker_velocity_gain=blocker_velocity_gain,
        blocker_gravity_compensation=blocker_gravity_compensation,
        blocker_home_offset=blocker_home_offset,
        blocker_target_peg_length_scale=blocker_target_peg_length_scale,
        blocker_return_position_gain=blocker_return_position_gain,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=True, record_metrics=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        initial_peg_to_hole = (
            obs["extra"]["peg_pose"][:, :3]
            - obs["extra"]["box_hole_pose"][:, :3]
        )
        initial_lateral = world_to_local(
            initial_peg_to_hole, obs["extra"]["box_hole_pose"][:, 3:],
        )[:, 1]
        maximum_lateral_shift = torch.zeros(
            num_envs, device=initial_lateral.device,
        )
        minimum_peg_z = obs["extra"]["peg_pose"][:, 2].clone()
        maximum_peg_xy_radius = torch.linalg.vector_norm(
            obs["extra"]["peg_pose"][:, :2], dim=1,
        )
        minimum_hole_distance = torch.full(
            (num_envs,), torch.inf, device=initial_lateral.device,
        )
        minimum_target_distance = torch.full_like(minimum_hole_distance, torch.inf)
        ever_engaged = torch.zeros(
            num_envs, dtype=torch.bool, device=initial_lateral.device,
        )
        ever_cleared = torch.zeros_like(ever_engaged)
        ever_constraint_violated = torch.zeros_like(ever_engaged)
        info = None
        action = torch.zeros(
            (num_envs,) + env.single_action_space.shape,
            device=initial_lateral.device,
        )
        for _ in range(steps):
            obs, _, _, _, info = env.step(action)
            peg_to_hole = (
                obs["extra"]["peg_pose"][:, :3]
                - obs["extra"]["box_hole_pose"][:, :3]
            )
            lateral = world_to_local(
                peg_to_hole, obs["extra"]["box_hole_pose"][:, 3:],
            )[:, 1]
            shift = (lateral - initial_lateral).abs()
            maximum_lateral_shift = torch.maximum(maximum_lateral_shift, shift)
            minimum_peg_z = torch.minimum(
                minimum_peg_z, obs["extra"]["peg_pose"][:, 2],
            )
            maximum_peg_xy_radius = torch.maximum(
                maximum_peg_xy_radius,
                torch.linalg.vector_norm(obs["extra"]["peg_pose"][:, :2], dim=1),
            )
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
            ever_constraint_violated |= info["constraint_violated"].bool()
        assert info is not None
        result = {
            "condition": condition,
            "episodes": num_envs,
            "finite_observation": bool(torch.isfinite(obs["extra"]["peg_pose"]).all()),
            "maximum_hole_frame_lateral_shift_mean": float(
                maximum_lateral_shift.mean()
            ),
            "ejection_observed_rate": float(
                (maximum_lateral_shift > 0.01).float().mean()
            ),
            "peg_below_floor_rate": float((minimum_peg_z < -0.02).float().mean()),
            "peg_out_of_bounds_rate": float(
                (maximum_peg_xy_radius > 0.8).float().mean()
            ),
            "blocker_engaged_rate": float(ever_engaged.float().mean()),
            "minimum_hole_distance_mean": float(minimum_hole_distance.mean()),
            "minimum_hole_distance_max": float(minimum_hole_distance.max()),
            "minimum_target_distance_mean": float(minimum_target_distance.mean()),
            "minimum_target_distance_max": float(minimum_target_distance.max()),
            "temporary_cleared_rate": float(ever_cleared.float().mean()),
            "constraint_violation_rate": float(
                ever_constraint_violated.float().mean()
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
    default_conditions = (
        "nominal",
        "positive_lateral_peg_ejection",
        "negative_lateral_peg_ejection",
        "permanent_hole_block",
        "temporary_hole_block",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=421000001)
    parser.add_argument("--ejection-force", type=float, default=1.7)
    parser.add_argument("--ejection-steps", type=int, default=5)
    parser.add_argument("--negative-ejection-force-scale", type=float, default=1.0)
    parser.add_argument("--ejection-target-displacement", type=float, default=0.0)
    parser.add_argument("--ejection-position-gain", type=float, default=80.0)
    parser.add_argument("--ejection-velocity-gain", type=float, default=4.0)
    parser.add_argument("--blocker-force", type=float, default=5.0)
    parser.add_argument("--blocker-position-gain", type=float, default=40.0)
    parser.add_argument("--blocker-velocity-gain", type=float, default=4.0)
    parser.add_argument("--blocker-gravity-compensation", type=float, default=0.12)
    parser.add_argument("--blocker-home-offset", type=float, default=0.05)
    parser.add_argument("--blocker-target-peg-length-scale", type=float, default=0.0)
    parser.add_argument("--blocker-return-position-gain", type=float, default=120.0)
    parser.add_argument(
        "--conditions", nargs="+", choices=default_conditions,
        default=list(default_conditions),
    )
    args = parser.parse_args()
    conditions = tuple(args.conditions)
    results = [
        run_condition(
            condition,
            args.num_envs,
            args.steps,
            args.seed + index,
            args.ejection_force,
            args.ejection_steps,
            args.negative_ejection_force_scale,
            args.ejection_target_displacement,
            args.ejection_position_gain,
            args.ejection_velocity_gain,
            args.blocker_force,
            args.blocker_position_gain,
            args.blocker_velocity_gain,
            args.blocker_gravity_compensation,
            args.blocker_home_offset,
            args.blocker_target_peg_length_scale,
            args.blocker_return_position_gain,
        )
        for index, condition in enumerate(conditions)
    ]
    print(json.dumps({"schema_version": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()
