"""D-100: live detouring is not specific to `master_chef_can`."""

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
from atr.language.goal_graph import Constraint, GoalGraph  # noqa: E402


def test_real_fetch_detours_around_second_protected_object_type():
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
        original_graph = env.unwrapped.goal_graph
        env.unwrapped.goal_graph = GoalGraph(
            instruction_text=(
                "Put the potted meat can and bowl on the table, and do not "
                "move the cracker box."
            ),
            goals=original_graph.goals,
            constraints=(
                Constraint(
                    id="dont_move_cracker_box",
                    kind="never_move",
                    target_object="cracker_box",
                    tolerance=0.05,
                ),
            ),
        )

        xs, ys, occupied = _get_or_build_grid(env)
        start = env.unwrapped.agent.base_link.pose.sp.p[:2]
        target = env.unwrapped._get_actor("potted_meat_can").pose.sp.p[:2]
        original_path = plan_path(xs, ys, occupied, start, target)
        assert original_path is not None

        route_point = np.asarray(original_path[len(original_path) // 2])
        protected = env.unwrapped._get_actor("cracker_box")
        dynamic = next(
            component
            for component in protected._objs[0].components
            if type(component).__name__ == "PhysxRigidDynamicComponent"
        )
        dynamic.kinematic = True
        protected.set_pose(sapien.Pose(p=[route_point[0], route_point[1], 0.5]))
        protected_before = protected.pose.sp.p.copy()
        env.unwrapped._initial_state = env.unwrapped._world_state()

        result = attempt_goal(
            env, env.unwrapped.goal_graph.goals[0], _TRAY_SLOTS[0],
        )
        displacement = float(np.linalg.norm(protected.pose.sp.p - protected_before))

        assert result["navigation_replanned"] is True
        assert result["predicted_affected_objects"] == ["cracker_box"]
        assert "blocked_reason" not in result
        assert result["achieved"], result["navigation_final_distance"]
        assert result["steps_used"] > 0
        assert displacement == 0.0
    finally:
        env.close()
