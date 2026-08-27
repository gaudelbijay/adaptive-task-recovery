#!/usr/bin/env python3
"""Contact-verified sweep of candidate Fetch grasp points on the YCB bowl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_replicacad_manipulation import (
    _arm_controller,
    _hold_gripper,
    _servo_arm_to,
    _world_to_local,
)
from atr.envs.tidy_up_replicacad_policies import _navigate_to, _yaw


# The bowl is wider than Fetch's gripper.  These candidates place one finger
# inside and one outside the rim instead of trying to pinch its full diameter.
_CANDIDATES = [
    (angle_degrees, height)
    for angle_degrees in range(0, 360, 45)
    for height in (0.0, 0.03)
]
_HIGH_RIM_CANDIDATES = [
    (lateral_offset, height)
    for height in (0.08, 0.12, 0.16)
    for lateral_offset in (0.0, 0.04, 0.08, -0.04)
]


def _face_target(env, target_xy, max_steps=80):
    used = 0
    for _ in range(max_steps):
        pose = env.unwrapped.agent.base_link.pose.sp
        delta = np.asarray(target_xy) - pose.p[:2]
        desired = float(np.arctan2(delta[1], delta[0]))
        error = (desired - _yaw(pose) + np.pi) % (2 * np.pi) - np.pi
        if abs(error) < 0.04:
            break
        action = np.zeros(9, dtype=np.float32)
        action[8] = np.clip(error / 0.3, -1, 1)
        env.step(action)
        used += 1
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--output", default="results/manipulation_ppo/bowl_grasp_sweep")
    args = parser.parse_args()
    high_rim_count = 3 * len(_HIGH_RIM_CANDIDATES)  # torso: lower, neutral, raise
    if not 0 <= args.candidate < 33 + high_rim_count:
        raise ValueError(
            f"candidate must be in [0, {32 + high_rim_count}]"
        )
    high_rim = args.candidate >= 33
    default_approach = high_rim or 16 <= args.candidate <= 18 or args.candidate >= 27
    if high_rim:
        angle_degrees = None
        candidate_within_torso_group = (args.candidate - 33) % len(_HIGH_RIM_CANDIDATES)
        lateral_offset, height = _HIGH_RIM_CANDIDATES[candidate_within_torso_group]
        offset = np.asarray([0.0, lateral_offset, height], dtype=float)
    elif default_approach:
        angle_degrees = None
        center_target = args.candidate == 17 or (args.candidate >= 27 and args.candidate % 2 == 1)
        offset = np.asarray([0.0, 0.0 if center_target else 0.07, 0.02], dtype=float)
    elif args.candidate >= 19:
        angle_degrees = 315
        offset = np.asarray([0.0, 0.0 if args.candidate == 19 else 0.07, 0.02], dtype=float)
    else:
        angle_degrees, offset_z = _CANDIDATES[args.candidate]
        offset = np.asarray([0.0, 0.0, offset_z], dtype=float)
    env = gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", intervention_kind="none",
    )
    try:
        env.reset(seed=0)
        obj = env.unwrapped._get_actor("bowl")
        if default_approach:
            approach_xy = None
            navigation = _navigate_to(env, obj.pose.sp.p, steps=250, target_object="bowl")
            face_steps = 0
        else:
            angle = np.deg2rad(angle_degrees)
            if args.candidate >= 21:
                approach_radius = (0.6, 0.5, 0.4)[(args.candidate - 21) // 2]
                offset[1] = 0.0 if args.candidate % 2 else 0.07
            else:
                approach_radius = 0.7
            approach_xy = obj.pose.sp.p[:2] + approach_radius * np.array([np.cos(angle), np.sin(angle)])
            navigation = _navigate_to(
                env, approach_xy, steps=300, target_object="bowl", distance_tol=0.18,
            )
            face_steps = _face_target(env, obj.pose.sp.p[:2])
        _hold_gripper(env, 1.0, 6)
        torso_direction = 0.0
        if 17 <= args.candidate < 45:
            torso_direction = -1.0
        elif args.candidate >= 57:
            torso_direction = 1.0
        if torso_direction:
            for _ in range(4):
                action = np.zeros(9, dtype=np.float32)
                action[3] = 1.0
                action[6] = torso_direction
                env.step(action)
        if 27 <= args.candidate < 33:
            advance_steps = (5, 10, 20)[(args.candidate - 27) // 2]
            for _ in range(advance_steps):
                action = np.zeros(9, dtype=np.float32)
                action[3] = 1.0
                action[7] = 0.3
                env.step(action)
        target = obj.pose.sp.p.astype(float) + offset
        arm = _arm_controller(env)
        target_local = _world_to_local(arm, target)
        used, distance = _servo_arm_to(env, arm, target_local, gripper_action=1.0)
        _hold_gripper(env, -1.0, 15)
        agent = env.unwrapped.agent
        left_force = agent.scene.get_pairwise_contact_forces(agent.finger1_link, obj)
        right_force = agent.scene.get_pairwise_contact_forces(agent.finger2_link, obj)
        left_transform = agent.finger1_link.pose.to_transformation_matrix()
        right_transform = agent.finger2_link.pose.to_transformation_matrix()
        grasped = bool(agent.is_grasping(obj))
        if grasped:
            lift_target = _arm_controller(env).ee_pose_at_base.p.numpy()[0] + np.array([0.0, 0.0, 0.15])
            _servo_arm_to(env, _arm_controller(env), lift_target, -1.0, max_steps=40)
        payload = {
            "candidate": args.candidate,
            "approach_angle_degrees": angle_degrees,
            "approach_xy": approach_xy.tolist() if approach_xy is not None else None,
            "face_steps": face_steps,
            "offset": offset.tolist(),
            "torso_direction": torso_direction,
            "target_world": target.tolist(),
            "target_local": np.asarray(target_local).tolist(),
            "root_position": arm.root_link.pose.sp.p.tolist(),
            "end_effector_local": arm.ee_pose_at_base.p.cpu().numpy()[0].tolist(),
            "navigation_reached": bool(navigation.reached_target),
            "servo_steps": int(used),
            "servo_distance": float(distance),
            "object_position": obj.pose.sp.p.tolist(),
            "end_effector_position": _arm_controller(env).ee_pose.sp.p.tolist(),
            "base_position": env.unwrapped.agent.base_link.pose.sp.p.tolist(),
            "base_yaw": _yaw(env.unwrapped.agent.base_link.pose.sp),
            "base_object_distance": float(np.linalg.norm(
                env.unwrapped.agent.base_link.pose.sp.p[:2] - obj.pose.sp.p[:2]
            )),
            "left_finger_position": agent.finger1_link.pose.sp.p.tolist(),
            "right_finger_position": agent.finger2_link.pose.sp.p.tolist(),
            "left_open_direction": (-left_transform[0, :3, 1]).cpu().tolist(),
            "right_open_direction": right_transform[0, :3, 1].cpu().tolist(),
            "left_contact_force": float(np.linalg.norm(left_force.cpu().numpy())),
            "right_contact_force": float(np.linalg.norm(right_force.cpu().numpy())),
            "grasped": grasped,
            "held_after_lift": bool(env.unwrapped.agent.is_grasping(obj)),
        }
    finally:
        env.close()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"candidate_{args.candidate:02d}.json"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
