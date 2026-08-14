"""D-096: live positive-detour validation in ReplicaCAD + Fetch."""

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
    attempt_goal,
)


def _make_live_detour_case():
    env = gym.make(
        "TidyUp-ReplicaCAD-v1",
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pos",
        intervention_kind="none",
    )
    env.reset(seed=0)
    xs, ys, occupied = _get_or_build_grid(env)
    start = env.unwrapped.agent.base_link.pose.sp.p[:2]
    target = env.unwrapped._get_actor("potted_meat_can").pose.sp.p[:2]
    original_path = plan_path(xs, ys, occupied, start, target)
    assert original_path is not None

    # Put the graph's already-protected object on a genuine waypoint of
    # the initially planned route. Kinematic mode makes this a stable
    # obstacle/constraint case rather than an unsupported object falling
    # under gravity before navigation begins.
    route_point = np.asarray(original_path[len(original_path) // 2])
    protected = env.unwrapped._get_actor("master_chef_can")
    dynamic = next(
        component
        for component in protected._objs[0].components
        if type(component).__name__ == "PhysxRigidDynamicComponent"
    )
    dynamic.kinematic = True
    protected.set_pose(sapien.Pose(p=[route_point[0], route_point[1], 0.5]))
    env.unwrapped._initial_state = env.unwrapped._world_state()
    return env, protected


def test_real_fetch_replans_around_protected_object_and_completes_goal():
    env, protected = _make_live_detour_case()
    try:
        protected_before = protected.pose.sp.p.copy()

        goal = env.unwrapped.goal_graph.goals[0]
        result = attempt_goal(env, goal, _TRAY_SLOTS[0])
        protected_after = protected.pose.sp.p.copy()

        assert result["navigation_safety_screened"] is True
        assert result["navigation_replanned"] is True
        assert result["navigation_reached_target"] is True
        assert result["predicted_affected_objects"] == ["master_chef_can"]
        assert "blocked_reason" not in result
        assert result["achieved"]
        assert result["steps_used"] > 0
        assert np.linalg.norm(protected_after - protected_before) == 0.0
    finally:
        env.close()


def test_replanning_recovers_goal_recall_that_stop_only_guard_loses():
    outcomes = {}
    for name, allow_replan in (("stop_only", False), ("replan", True)):
        env, protected = _make_live_detour_case()
        try:
            protected_before = protected.pose.sp.p.copy()
            goal = env.unwrapped.goal_graph.goals[0]
            result = attempt_goal(
                env, goal, _TRAY_SLOTS[0], allow_replan=allow_replan,
            )
            outcomes[name] = {
                **result,
                "protected_displacement": float(
                    np.linalg.norm(protected.pose.sp.p - protected_before)
                ),
            }
        finally:
            env.close()

    assert outcomes["stop_only"]["achieved"] is False
    assert outcomes["stop_only"]["skipped"] is True
    assert outcomes["stop_only"]["steps_used"] == 0
    assert "dont_move_master_chef_can" in outcomes["stop_only"]["blocked_reason"]

    assert outcomes["replan"]["achieved"]
    assert outcomes["replan"]["skipped"] is False
    assert outcomes["replan"]["navigation_replanned"] is True
    assert outcomes["replan"]["navigation_reached_target"] is True
    assert outcomes["replan"]["steps_used"] > 0

    assert outcomes["stop_only"]["protected_displacement"] == 0.0
    assert outcomes["replan"]["protected_displacement"] == 0.0
