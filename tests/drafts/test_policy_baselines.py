"""Tests for the static-vs-feasibility-aware policy comparison — the first
runnable test of H2 (docs/01): feasibility-conditioning beats a static
policy after an irreversible change.
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUp-v1)
from atr.language.goal_graph import dependent_goals_example  # noqa: E402
from task_schema_draft.policy_baselines import feasibility_aware_policy, static_policy  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestPolicyComparisonAfterBowlDestroyed:
    def test_both_achieve_the_still_feasible_goal(self):
        for policy in (static_policy, feasibility_aware_policy):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                result = policy(env)
                assert result["goals_achieved"] == 1
                assert result["per_goal"]["place_mug"]["achieved"]
                assert not result["per_goal"]["place_bowl"]["achieved"]
            finally:
                env.close()

    def test_static_policy_wastes_steps_on_the_destroyed_goal(self):
        env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = static_policy(env)
            assert result["per_goal"]["place_bowl"]["skipped"] is False
            assert result["wasted_steps"] > 0
        finally:
            env.close()

    def test_feasibility_aware_policy_skips_instead_of_wasting(self):
        env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = feasibility_aware_policy(env)
            assert result["per_goal"]["place_bowl"]["skipped"] is True
            assert result["wasted_steps"] == 0
        finally:
            env.close()

    def test_feasibility_aware_uses_fewer_total_steps_for_the_same_outcome(self):
        """The H2 comparison: same goals achieved, less wasted effort."""
        results = {}
        for name, policy in [("static", static_policy), ("feasibility_aware", feasibility_aware_policy)]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()
        assert results["static"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["feasibility_aware"]["total_steps"] < results["static"]["total_steps"]


class TestGoalDependencyGating:
    """D-037, resolving the D-013 review's open question 3 with a real
    live-env demonstration, not just a pure-function test: place_bowl
    (dependent_goals_example()) depends_on place_mug. Directly removes
    red_mug the same way tidy_up_env.py's own bowl_destroyed intervention
    removes blue_bowl (see that file's _trigger_intervention) -- no new
    intervention_kind needed for this."""

    def test_dependent_goal_blocked_when_its_prerequisite_can_never_complete(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            env.unwrapped._objects["red_mug"].remove_from_scene()
            env.unwrapped._exists["red_mug"] = False
            result = feasibility_aware_policy(env, graph=dependent_goals_example())
        finally:
            env.close()
        # place_bowl's own target (blue_bowl) still exists -- goal_feasible()
        # alone would say yes. It's blocked because place_mug, its
        # depends_on prerequisite, can never be achieved (red_mug is gone).
        assert result["per_goal"]["place_mug"]["achieved"] is False
        assert result["per_goal"]["place_bowl"]["skipped"] is True
        assert result["per_goal"]["place_bowl"]["achieved"] is False
        assert result["wasted_steps"] == 0

    def test_dependent_goal_proceeds_once_its_prerequisite_is_achieved(self):
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            result = feasibility_aware_policy(env, graph=dependent_goals_example())
        finally:
            env.close()
        assert result["per_goal"]["place_mug"]["achieved"]
        assert result["per_goal"]["place_bowl"]["skipped"] is False
        assert result["per_goal"]["place_bowl"]["achieved"]


class TestPolicyComparisonNoIntervention:
    def test_both_achieve_both_goals_when_nothing_is_destroyed(self):
        for policy in (static_policy, feasibility_aware_policy):
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=0)
                result = policy(env)
                assert result["goals_achieved"] == 2
                assert result["wasted_steps"] == 0
            finally:
                env.close()
