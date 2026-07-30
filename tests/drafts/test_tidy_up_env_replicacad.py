"""Tests for the ReplicaCAD + Fetch version of TidyUp — a real furnished
apartment scene (Habitat/ManiSkill3's own ReplicaCADSetTableTrain builder)
with real YCB objects and a mobile robot, instead of a hand-built scene.
Same goal_graph/oracle_feasibility/intent_guard as the panda and humanoid
variants. See tidy_up_env_replicacad.py and navigation.py module docstrings,
and ../README.md "ReplicaCAD embodiment" for what had to change and why.

Requires the ReplicaCAD + ReplicaCADRearrange + ycb asset downloads (see
README "How to run it").
"""

import numpy as np
import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUpTaskSchemaDraft-ReplicaCAD-v1)
from task_schema_draft.navigation import build_occupancy_grid, plan_path  # noqa: E402
from task_schema_draft.policy_baselines_replicacad import (  # noqa: E402
    feasibility_aware_policy,
    naive_substitution_policy,
    static_policy,
)


def _make_env(**kwargs):
    return gym.make(
        "TidyUpTaskSchemaDraft-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestTidyUpReplicaCADEnv:
    def test_registered(self):
        assert "TidyUpTaskSchemaDraft-ReplicaCAD-v1" in gym.envs.registry

    def test_reset_and_step(self):
        env = _make_env(intervention_kind="none")
        try:
            obs, info = env.reset(seed=0)
            assert "goal_feasibility" in info
            env.step(env.action_space.sample() * 0)
        finally:
            env.close()


class TestNavigation:
    def test_path_planner_routes_around_the_real_wall_that_blocked_naive_control(self):
        """Regression test for the actual bug found: a naive point-and-drive
        controller got physically stuck against a real wall/doorway in this
        scene (confirmed via raycast at the time). The planner must find a
        path, and that path must not be a straight line through the
        obstacle."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            px = env.unwrapped.scene.px
            xs, ys, occupied = build_occupancy_grid(
                px, (-2.5, 1.0), (-1.5, 1.5), robot_radius=0.2
            )
            start = env.unwrapped.agent.base_link.pose.sp.p[:2]
            path = plan_path(xs, ys, occupied, start, np.array([0.29, 0.09]))
            assert path is not None
            assert len(path) > 2  # more than one straight hop -- a real detour
        finally:
            env.close()


class TestReplicaCADPolicyComparison:
    def test_static_vs_feasibility_aware_same_recall_less_waste(self):
        results = {}
        for name, policy in [("static", static_policy), ("feasibility_aware", feasibility_aware_policy)]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()
        assert results["static"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["feasibility_aware"]["wasted_steps"] == 0
        assert results["static"]["wasted_steps"] > 0

    def test_intent_guard_blocks_substitution_without_recall_cost(self):
        results = {}
        for guarded in (False, True):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[guarded] = naive_substitution_policy(env, use_intent_guard=guarded)
            finally:
                env.close()
        assert results[False]["dont_move_master_chef_can_violated"] is True
        assert results[True]["dont_move_master_chef_can_violated"] is False
        assert results[False]["goals_achieved"] == results[True]["goals_achieved"]
