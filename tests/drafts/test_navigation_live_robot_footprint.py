"""D-103: safety screening uses Fetch's real collision-mesh footprint."""

import numpy as np
import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402

import task_schema_draft  # noqa: E402, F401
from atr.envs.navigation import plan_path  # noqa: E402
from atr.envs.tidy_up_replicacad_policies import (  # noqa: E402
    _TRAY_SLOTS,
    _get_or_build_grid,
    _object_planar_radii,
    _robot_planar_radius,
    attempt_goal,
)


def test_real_fetch_footprint_detects_overlap_beyond_old_grid_margin():
    env = gym.make(
        "TidyUp-ReplicaCAD-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pos",
        intervention_kind="none",
    )
    try:
        env.reset(seed=0)
        xs, ys, occupied = _get_or_build_grid(env)
        start = env.unwrapped.agent.base_link.pose.sp.p[:2]
        target = env.unwrapped._get_actor("potted_meat_can").pose.sp.p[:2]
        original_path = plan_path(xs, ys, occupied, start, target)
        assert original_path is not None

        state = env.unwrapped._world_state()
        robot_radius = _robot_planar_radius(env)
        object_radius = _object_planar_radii(env, state)["master_chef_can"]
        assert 0.27 < robot_radius < 0.30

        segment_index = next(
            i
            for i, (a, b) in enumerate(zip(original_path, original_path[1:]))
            if abs(a[1] - b[1]) < 1e-8
        )
        midpoint = (np.asarray(original_path[segment_index]) + np.asarray(original_path[segment_index + 1])) / 2
        old_limit = 0.2 + object_radius
        measured_limit = robot_radius + object_radius
        offset = (old_limit + measured_limit) / 2
        protected_xy = midpoint + np.array([0.0, offset])

        protected = env.unwrapped._get_actor("master_chef_can")
        dynamic = next(
            component
            for component in protected._objs[0].components
            if type(component).__name__ == "PhysxRigidDynamicComponent"
        )
        dynamic.kinematic = True
        protected.set_pose(sapien.Pose(p=[protected_xy[0], protected_xy[1], 0.05]))
        protected_before = protected.pose.sp.p.copy()
        env.unwrapped._initial_state = env.unwrapped._world_state()

        result = attempt_goal(
            env,
            env.unwrapped.goal_graph.goals[0],
            _TRAY_SLOTS[0],
            robot_clearance_radius=robot_radius,
        )
        displacement = float(np.linalg.norm(protected.pose.sp.p - protected_before))

        assert offset > old_limit
        assert offset < measured_limit
        assert result["navigation_replanned"] is True
        assert result["predicted_affected_objects"] == ["master_chef_can"]
        assert "blocked_reason" not in result
        assert result["achieved"]
        assert displacement == 0.0
    finally:
        env.close()
