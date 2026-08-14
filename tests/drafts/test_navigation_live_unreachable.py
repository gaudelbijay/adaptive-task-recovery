"""D-109/D-110: real no-route failure propagates without side effects."""

import numpy as np
import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402
import sapien  # noqa: E402
from scipy.ndimage import label  # noqa: E402

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_replicacad_policies import (  # noqa: E402
    _TRAY_SLOTS,
    _get_or_build_grid,
    attempt_goal,
)
from atr.envs.navigation import _nearest_free_cell  # noqa: E402
from atr.language.goal_graph import Goal  # noqa: E402
from atr.policies.baselines import _summarize  # noqa: E402


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_real_unreachable_attempt_has_no_side_effects_and_is_counted(seed):
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
        env.reset(seed=seed)
        actor = env.unwrapped._get_actor("master_chef_can")
        object_before = actor.pose.sp.p.copy()
        base_before = env.unwrapped.agent.base_link.pose.sp.p.copy()
        steps_before = env.unwrapped._elapsed_control_steps

        result = attempt_goal(
            env,
            Goal(
                id="place_unreachable_can",
                predicate="on_tray",
                target_object="master_chef_can",
            ),
            _TRAY_SLOTS[0],
        )

        base_after = env.unwrapped.agent.base_link.pose.sp.p.copy()
        object_after = actor.pose.sp.p.copy()
        summary = _summarize({"place_unreachable_can": result})

        assert result["navigation_failure_reason"] == (
            "unreachable: no collision-free grid path"
        )
        assert result["navigation_failed"] is True
        assert result["navigation_reached_target"] is False
        assert result["achieved"] is False
        assert result["skipped"] is False
        assert result["steps_used"] == 0
        assert summary["navigation_failures"] == 1
        assert summary["navigation_safety_blocks"] == 0
        assert env.unwrapped._elapsed_control_steps == steps_before
        assert np.array_equal(base_after, base_before)
        assert np.array_equal(object_after, object_before)
    finally:
        env.close()


def test_real_unreachable_result_follows_geometry_not_object_name():
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
        original = env.unwrapped._get_actor("master_chef_can")
        alternate = env.unwrapped._get_actor("cracker_box")
        original_pose = original.pose.sp.p.copy()
        alternate_pose = alternate.pose.sp.p.copy()

        # Swap the actors so a different semantic object occupies the same
        # disconnected geometric region, without adding another obstacle.
        original.set_pose(sapien.Pose(p=alternate_pose))
        alternate.set_pose(sapien.Pose(p=original_pose))
        env.unwrapped._initial_state = env.unwrapped._world_state()
        base_before = env.unwrapped.agent.base_link.pose.sp.p.copy()
        object_before = alternate.pose.sp.p.copy()
        steps_before = env.unwrapped._elapsed_control_steps

        result = attempt_goal(
            env,
            Goal(
                id="place_unreachable_box",
                predicate="on_tray",
                target_object="cracker_box",
            ),
            _TRAY_SLOTS[0],
        )
        summary = _summarize({"place_unreachable_box": result})

        assert result["navigation_failure_reason"] == (
            "unreachable: no collision-free grid path"
        )
        assert result["navigation_failed"] is True
        assert result["achieved"] is False
        assert result["steps_used"] == 0
        assert summary["navigation_failures"] == 1
        assert summary["navigation_safety_blocks"] == 0
        assert env.unwrapped._elapsed_control_steps == steps_before
        assert np.array_equal(env.unwrapped.agent.base_link.pose.sp.p, base_before)
        assert np.array_equal(alternate.pose.sp.p, object_before)
    finally:
        env.close()


def test_real_unreachable_result_holds_in_a_second_disconnected_region():
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
        components, _ = label(~occupied, structure=np.ones((3, 3), dtype=int))
        start_cell = _nearest_free_cell(
            xs, ys, occupied, env.unwrapped.agent.base_link.pose.sp.p[:2]
        )
        original_cell = _nearest_free_cell(
            xs,
            ys,
            occupied,
            env.unwrapped._get_actor("master_chef_can").pose.sp.p[:2],
        )
        excluded = {0, components[start_cell], components[original_cell]}
        candidates = [
            (int(np.count_nonzero(components == component)), component)
            for component in np.unique(components)
            if component not in excluded
        ]
        _, second_component = max(candidates)
        target_cell = tuple(np.argwhere(components == second_component)[0])
        target_position = np.array(
            [xs[target_cell[0]], ys[target_cell[1]], 0.5], dtype=float
        )

        actor = env.unwrapped._get_actor("cracker_box")
        actor.set_pose(sapien.Pose(p=target_position))
        env.unwrapped._initial_state = env.unwrapped._world_state()
        base_before = env.unwrapped.agent.base_link.pose.sp.p.copy()
        object_before = actor.pose.sp.p.copy()
        steps_before = env.unwrapped._elapsed_control_steps

        result = attempt_goal(
            env,
            Goal(
                id="place_box_from_second_region",
                predicate="on_tray",
                target_object="cracker_box",
            ),
            _TRAY_SLOTS[0],
        )
        summary = _summarize({"place_box_from_second_region": result})

        assert second_component != components[original_cell]
        assert result["navigation_failure_reason"] == (
            "unreachable: no collision-free grid path"
        )
        assert result["steps_used"] == 0
        assert result["achieved"] is False
        assert summary["navigation_failures"] == 1
        assert summary["navigation_safety_blocks"] == 0
        assert env.unwrapped._elapsed_control_steps == steps_before
        assert np.array_equal(env.unwrapped.agent.base_link.pose.sp.p, base_before)
        assert np.array_equal(actor.pose.sp.p, object_before)
    finally:
        env.close()
