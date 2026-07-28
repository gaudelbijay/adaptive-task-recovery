"""Manipulation-skill + RGB-D observation spike.

Tests the two remaining untested rows from docs/04-benchmark-environment.md's
selection requirements (see D-010, D-011 in ai-notes/decisions.md):

1. Can we get RGB-D observations out of ManiSkill3? (`rgbd_obs_shapes`)
2. Does a "reusable reach/grasp" skill actually work on this machine?
   (`scripted_pick_and_lift`)

ManiSkill3 ships canned motion-planning solutions for this
(`mani_skill.examples.motionplanning.panda.solutions`), but they depend on
`mplib`, which fails to build from source on this machine (Apple Silicon
macOS, Python 3.12 — pins `libclang==11.0.1`, no matching wheel). Instead,
this uses ManiSkill3's built-in Cartesian end-effector controller
(`pd_ee_delta_pos`, IK via `pinocchio` — installable here as the `pin` pip
package) with a simple hand-scripted waypoint sequence: no collision-aware
path planning, just proportional control toward a few fixed targets. That's
a lower bar than full motion planning, but it's enough to test whether basic
reach/grasp actuation works at all.
"""

from __future__ import annotations

import numpy as np


def rgbd_obs_shapes(env) -> dict:
    """Reset a rgbd-obs-mode env and return sensor shapes/dtypes/ranges."""
    obs, _ = env.reset(seed=0)
    camera_name = next(iter(obs["sensor_data"].keys()))
    sensor = obs["sensor_data"][camera_name]
    return {
        "camera_name": camera_name,
        "rgb_shape": tuple(sensor["rgb"].shape),
        "rgb_dtype": str(sensor["rgb"].dtype),
        "rgb_range": (sensor["rgb"].float().min().item(), sensor["rgb"].float().max().item()),
        "depth_shape": tuple(sensor["depth"].shape),
        "depth_dtype": str(sensor["depth"].dtype),
        "depth_range": (
            sensor["depth"].float().min().item(),
            sensor["depth"].float().max().item(),
        ),
    }


def _go_to(env, target_xyz: np.ndarray, gripper: float, steps: int, tol: float = 0.005):
    """Proportional control toward a fixed Cartesian target. `pd_ee_delta_pos`
    maps action in [-1, 1] to a delta of +/-0.1m per control step."""
    for _ in range(steps):
        tcp = env.unwrapped.agent.tcp.pose.sp.p
        delta = np.clip((target_xyz - tcp) / 0.1, -1, 1)
        action = np.array([delta[0], delta[1], delta[2], gripper], dtype=np.float32)
        env.step(action)
        if np.linalg.norm(target_xyz - env.unwrapped.agent.tcp.pose.sp.p) < tol:
            break


def scripted_pick_and_lift(env, seed: int, lift_success_margin: float = 0.08) -> bool:
    """Reach above the cube, descend, close the gripper, lift. Returns
    whether the cube ended up at least `lift_success_margin` above its
    start height (env must be PickCube-v1 with control_mode='pd_ee_delta_pos')."""
    env.reset(seed=seed)
    cube_start_z = env.unwrapped.cube.pose.sp.p[2]
    cube_xyz = env.unwrapped.cube.pose.sp.p.copy()

    _go_to(env, cube_xyz + np.array([0, 0, 0.08]), gripper=1.0, steps=40)   # approach, open
    _go_to(env, cube_xyz + np.array([0, 0, 0.005]), gripper=1.0, steps=30)  # descend, open
    _go_to(env, cube_xyz + np.array([0, 0, 0.005]), gripper=-1.0, steps=15)  # close
    lift_target = env.unwrapped.agent.tcp.pose.sp.p + np.array([0, 0, 0.15])
    _go_to(env, lift_target, gripper=-1.0, steps=30)  # lift

    cube_final_z = env.unwrapped.cube.pose.sp.p[2]
    return bool(cube_final_z > cube_start_z + lift_success_margin)
