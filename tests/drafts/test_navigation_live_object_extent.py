"""D-102: production navigation uses real collision-mesh object extents."""

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
    attempt_goal,
)


def test_real_collision_extent_triggers_detour_when_center_is_outside_clearance():
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

        protected = env.unwrapped._get_actor("master_chef_can")
        radius = _object_planar_radii(
            env, env.unwrapped._world_state(),
        )["master_chef_can"]
        assert 0.02 < radius < 0.10

        # Offset perpendicular to a horizontal route segment. The center is
        # outside the 0.2 m robot clearance, but the real collision mesh
        # overlaps it: 0.2 < offset < 0.2 + object radius.
        segment_index = next(
            i
            for i, (a, b) in enumerate(zip(original_path, original_path[1:]))
            if abs(a[1] - b[1]) < 1e-8
        )
        a = np.asarray(original_path[segment_index])
        b = np.asarray(original_path[segment_index + 1])
        midpoint = (a + b) / 2
        offset = 0.2 + radius / 2
        protected_xy = midpoint + np.array([0.0, offset])

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
            env, env.unwrapped.goal_graph.goals[0], _TRAY_SLOTS[0],
        )
        displacement = float(np.linalg.norm(protected.pose.sp.p - protected_before))

        assert result["navigation_replanned"] is True
        assert result["predicted_affected_objects"] == ["master_chef_can"]
        assert "blocked_reason" not in result
        assert result["achieved"]
        assert displacement == 0.0
    finally:
        env.close()
