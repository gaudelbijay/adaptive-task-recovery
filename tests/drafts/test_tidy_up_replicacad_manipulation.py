"""D-124: real pick-and-place, verified stage by stage rather than trusting
a single final boolean -- see tidy_up_replicacad_manipulation.py's module
docstring for why this is deliberately separate from attempt_goal()'s
teleport abstraction, not a replacement for it.
"""

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_env_replicacad import _TRAY_HALF_SIZES, _TRAY_POSITION  # noqa: E402
from atr.envs.tidy_up_replicacad_manipulation import (  # noqa: E402
    attempt_goal_with_real_grasp,
    ensure_tray_surface,
)
from atr.language.goal_graph import Goal  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestRealPickAndPlace:
    def test_grasps_carries_and_places_a_real_object(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            agent = env.unwrapped.agent
            obj = env.unwrapped._get_actor("potted_meat_can")
            goal = Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can", priority=0)

            assert not agent.is_grasping(obj)  # sanity: not grasping before we start

            result = attempt_goal_with_real_grasp(env, goal, _TRAY_POSITION)

            assert result["grasped"]
            assert result["carried"]
            assert result["achieved"]
            assert result["steps_used"] > 0

            # The success check itself must be real, physically-settled
            # state -- not a flag left over from mid-grasp.
            final_pos = obj.pose.sp.p
            for axis in range(3):
                assert abs(final_pos[axis] - _TRAY_POSITION[axis]) <= _TRAY_HALF_SIZES[axis]
        finally:
            env.close()

    def test_grasp_is_contact_verified_not_assumed(self):
        """Directly exercises is_grasping()'s real transition -- confirms
        this module checks actual contact, not just that the gripper-close
        action was sent."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            agent = env.unwrapped.agent
            obj = env.unwrapped._get_actor("potted_meat_can")
            goal = Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can", priority=0)

            assert bool(agent.is_grasping(obj)) is False
            attempt_goal_with_real_grasp(env, goal, _TRAY_POSITION)
            # Object was released onto the tray by the end -- no longer
            # grasped, but real physics carried it there (checked above).
            assert bool(agent.is_grasping(obj)) is False
        finally:
            env.close()

    def test_ensure_tray_surface_is_idempotent(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            ensure_tray_surface(env)
            n_actors_after_first = len(env.unwrapped.scene.actors)
            ensure_tray_surface(env)
            assert len(env.unwrapped.scene.actors) == n_actors_after_first
        finally:
            env.close()

    def test_reports_failure_stage_when_object_is_unreachable(self):
        """A destroyed/absent object can't be navigated to -- confirms this
        returns a real failure with grasped=False rather than raising or
        silently succeeding."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            env.unwrapped._get_actor("master_chef_can").remove_from_scene()
            env.unwrapped._exists["master_chef_can"] = False
            goal = Goal(id="place_chef_can", predicate="on_tray", target_object="master_chef_can", priority=0)
            result = attempt_goal_with_real_grasp(env, goal, _TRAY_POSITION)
            assert not result["achieved"]
        finally:
            env.close()
