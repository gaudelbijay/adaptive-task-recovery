#!/usr/bin/env python3
"""Sweep 6-DoF top-down grasp poses for the ReplicaCAD YCB bowl."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien
from scipy.spatial.transform import Rotation

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_replicacad_manipulation import _arm_controller, _hold_gripper
from atr.envs.tidy_up_replicacad_policies import _navigate_to


_CANDIDATES = [
    (angle, radial_offset)
    for angle in (0, 45, 90, 135)
    for radial_offset in (0.0, 0.05, 0.08)
]


def _quat_wxyz_to_rotvec(quaternion):
    w, x, y, z = quaternion
    return Rotation.from_quat([x, y, z, w]).as_rotvec()


def _pose_action(env, action):
    env.step({"control_mode": "pd_ee_delta_pose", "action": action[None, :]})


def _servo_pose(env, target_world_pose, gripper, max_steps=120):
    agent = env.unwrapped.agent
    # Create/switch the controller before obtaining its root/EE frames.
    zero = np.zeros(12, dtype=np.float32)
    zero[6] = gripper
    _pose_action(env, zero)
    ctrl = agent.controller.controllers["arm"]
    pos_error = rot_error = float("inf")
    for step in range(max_steps):
        target_local = ctrl.root_link.pose.sp.inv() * target_world_pose
        current_local = ctrl.ee_pose_at_base.sp
        delta = target_local * current_local.inv()
        pos_error = float(np.linalg.norm(delta.p))
        rotvec = _quat_wxyz_to_rotvec(delta.q)
        rot_error = float(np.linalg.norm(rotvec))
        if pos_error < 0.025 and rot_error < 0.08:
            return step, pos_error, rot_error
        action = np.zeros(12, dtype=np.float32)
        action[:3] = np.clip(delta.p / 0.1, -1, 1)
        # Fetch's controller maps normalized [-1, 1] rotation actions onto a
        # per-step axis-angle range of [-0.1, 0.1] rad. The former diagnostic
        # divided by 2*pi and clipped to 0.01, commanding only ~0.001 rad and
        # never reaching any requested orientation within the sweep budget.
        # Keep each IK request small. Full normalized commands ask for a
        # 0.1-rad multi-axis jump, which this low-counter configuration rejects
        # and silently holds at the prior qpos.
        action[3:6] = np.clip(rotvec / 0.1, -0.1, 0.1)
        action[6] = gripper
        _pose_action(env, action)
    return max_steps, pos_error, rot_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--output", default="results/manipulation_ppo/bowl_pose_sweep")
    args = parser.parse_args()
    closing_angle, radial_offset = _CANDIDATES[args.candidate]
    env = gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", intervention_kind="none",
    )
    try:
        env.reset(seed=0)
        agent = env.unwrapped.agent
        obj = env.unwrapped._get_actor("bowl")
        navigation = _navigate_to(env, obj.pose.sp.p, steps=250, target_object="bowl")
        _hold_gripper(env, 1.0, 6)
        # Lower the torso to put the low countertop inside the arm workspace.
        for _ in range(4):
            action = np.zeros(9, dtype=np.float32)
            action[3], action[6] = 1.0, -1.0
            env.step(action)
        theta = np.deg2rad(closing_angle)
        closing = np.array([np.cos(theta), np.sin(theta), 0.0])
        approaching = np.array([0.0, 0.0, -1.0])
        center = obj.pose.sp.p.astype(float).copy()
        center[:2] += radial_offset * closing[:2]
        center[2] += 0.055
        grasp_pose = agent.build_grasp_pose(approaching, closing, center)
        current_ee = _arm_controller(env).ee_pose.sp
        orientation_pose = sapien.Pose(p=current_ee.p, q=grasp_pose.q)
        orientation_steps, orientation_pos_error, orientation_rot_error = _servo_pose(
            env, orientation_pose, gripper=1.0, max_steps=400,
        )
        steps, pos_error, rot_error = _servo_pose(env, grasp_pose, gripper=1.0)
        close_action = np.zeros(12, dtype=np.float32)
        close_action[6] = -1.0
        for _ in range(20):
            _pose_action(env, close_action)
        left = agent.scene.get_pairwise_contact_forces(agent.finger1_link, obj)
        right = agent.scene.get_pairwise_contact_forces(agent.finger2_link, obj)
        grasped = bool(agent.is_grasping(obj))
        if grasped:
            lift_pose = agent.controller.controllers["arm"].ee_pose.sp
            lift_pose.set_p(lift_pose.p + np.array([0.0, 0.0, 0.15]))
            _servo_pose(env, lift_pose, gripper=-1.0, max_steps=60)
        payload = {
            "candidate": args.candidate,
            "closing_angle_degrees": closing_angle,
            "radial_offset": radial_offset,
            "navigation_reached": bool(navigation.reached_target),
            "orientation_steps": orientation_steps,
            "orientation_position_error": orientation_pos_error,
            "orientation_rotation_error": orientation_rot_error,
            "servo_steps": steps,
            "position_error": pos_error,
            "rotation_error": rot_error,
            "left_contact_force": float(np.linalg.norm(left.cpu().numpy())),
            "right_contact_force": float(np.linalg.norm(right.cpu().numpy())),
            "grasped": grasped,
            "held_after_lift": bool(agent.is_grasping(obj)),
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
