"""Tests for atr.policies.symbolic_replanner (D-067) -- docs/10's
"symbolic replanner with learned state" required baseline.

TestPlan is pure-function (no simulator) on purpose -- runs in the
fast-checks CI tier, same reason test_splits.py/test_intent_guard.py's
TestValidateAction do. TestRunReplannerEpisode needs mani_skill (a real
env) and, for the learned-state case, a real CLIP forward pass.
"""

from atr.language.goal_graph import canonical_example, dependent_goals_example
from atr.policies.symbolic_replanner import plan


class TestPlan:
    """dependent_goals_example()'s real ordering constraint (place_bowl,
    priority 1, depends on place_mug, priority 0, being *achieved*) is
    the actual test of genuine planning here -- a fixed tuple-order walk
    gets the right *answer* by coincidence (mug already comes first in
    the tuple); these tests check the *planner* reasons about it, not
    just that the final per_goal result happens to look right."""

    def test_orders_the_prerequisite_before_the_dependent_goal(self):
        graph = dependent_goals_example()
        result = plan(graph, {"red_mug": True, "blue_bowl": True})
        assert [g.id for g in result] == ["place_mug", "place_bowl"]

    def test_excludes_the_dependent_goal_when_only_it_is_infeasible(self):
        graph = dependent_goals_example()
        result = plan(graph, {"red_mug": True, "blue_bowl": False})
        assert [g.id for g in result] == ["place_mug"]

    def test_prerequisite_infeasible_cascades_to_the_dependent_goal_too(self):
        """The real planning case a fixed pass can't express at all:
        place_bowl isn't merely infeasible on its own -- it can never be
        achieved this episode because its prerequisite can't be, even
        though blue_bowl itself still exists."""
        graph = dependent_goals_example()
        result = plan(graph, {"red_mug": False, "blue_bowl": True})
        assert result == []

    def test_no_dependency_graph_plans_every_feasible_goal(self):
        graph = canonical_example()
        result = plan(
            graph,
            {"red_mug": True, "blue_bowl": False, "glass": True, "medicine_bottle": True},
        )
        assert [g.id for g in result] == ["place_mug"]

    def test_already_achieved_goals_are_excluded_from_a_fresh_plan(self):
        graph = dependent_goals_example()
        result = plan(graph, {"red_mug": True, "blue_bowl": True}, achieved_ids=frozenset({"place_mug"}))
        assert [g.id for g in result] == ["place_bowl"]


import gymnasium as gym  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_replicacad_humanoid_policies import _TRAY_SLOTS, attempt_goal  # noqa: E402
from atr.pipeline import _instruction_graph  # noqa: E402
from atr.policies.symbolic_replanner import run_replanner_episode  # noqa: E402

_GRAPH = _instruction_graph()


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
    )


def _oracle_exists_fn(env):
    return dict(env.unwrapped._exists)


class TestRunReplannerEpisodeWithPrivilegedState:
    def test_matches_oracle_after_chef_can_destroyed(self):
        env = _make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = run_replanner_episode(env, _GRAPH, attempt_goal, _TRAY_SLOTS, _oracle_exists_fn)
            oracle_exists = dict(env.unwrapped._exists)
        finally:
            env.close()

        assert oracle_exists["master_chef_can"] is False
        assert oracle_exists["potted_meat_can"] is True
        assert result["per_goal"]["place_potted_meat_can"]["achieved"]
        assert result["per_goal"]["place_master_chef_can"]["skipped"] is True
        assert result["wasted_steps"] == 0


class TestRunReplannerEpisodeWithLearnedState:
    """The "learned state" half of this baseline's name: exists_fn reads
    a real rendered frame through CLIP (`visual_object_exists()`, D-020)
    instead of privileged state -- the same kind of substitution
    test_pipeline.py/test_clip_feasibility.py already exercise for the
    other policies, now for a genuine planner instead of a fixed pass."""

    def test_matches_oracle_using_clip_instead_of_privileged_state(self):
        pytest.importorskip("open_clip")
        from atr.feasibility.clip_feasibility import visual_object_exists

        def clip_exists_fn(env):
            frame = env.render()[0].cpu().numpy()
            return {
                "potted_meat_can": visual_object_exists(frame, "potted_meat_can"),
                "master_chef_can": visual_object_exists(frame, "master_chef_can"),
            }

        env = _make_env(intervention_kind="chef_can_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = run_replanner_episode(env, _GRAPH, attempt_goal, _TRAY_SLOTS, clip_exists_fn)
            oracle_exists = dict(env.unwrapped._exists)
        finally:
            env.close()

        assert oracle_exists["master_chef_can"] is False
        assert oracle_exists["potted_meat_can"] is True
        assert result["per_goal"]["place_potted_meat_can"]["achieved"]
        assert result["per_goal"]["place_master_chef_can"]["skipped"] is True
        assert result["wasted_steps"] == 0
