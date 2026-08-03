"""Tests for the intent guard — first toy test of H3 (docs/01): explicit
constraint checking reduces violations without collapsing goal recall.
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUpTaskSchemaDraft-v1)
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.constraints.intent_guard import validate_action  # noqa: E402
from task_schema_draft.policy_baselines import naive_substitution_policy  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "TidyUpTaskSchemaDraft-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestValidateAction:
    def test_blocks_moving_a_never_move_object_that_isnt_a_goal_target(self):
        graph = canonical_example()
        allowed, reason = validate_action("glass", graph)
        assert allowed is False
        assert "dont_move_glass" in reason

    def test_allows_moving_an_actual_goal_target(self):
        graph = canonical_example()
        allowed, _ = validate_action("red_mug", graph)
        assert allowed is True

    def test_allows_objects_with_no_constraint_at_all(self):
        graph = canonical_example()
        allowed, _ = validate_action("blue_bowl", graph)
        assert allowed is True


class TestNaiveSubstitutionPolicy:
    def test_unguarded_violates_the_constraint(self):
        env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = naive_substitution_policy(env, use_intent_guard=False)
            assert result["substitution_attempted"] is True
            assert result["dont_move_glass_violated"] is True
        finally:
            env.close()

    def test_guarded_blocks_the_substitution_and_avoids_the_violation(self):
        env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
        try:
            env.reset(seed=0)
            result = naive_substitution_policy(env, use_intent_guard=True)
            assert result["substitution_attempted"] is False
            assert result["dont_move_glass_violated"] is False
        finally:
            env.close()

    def test_guard_costs_zero_recall_in_this_scenario(self):
        """The substitution never legitimately counted toward place_bowl
        either way (the real bowl still doesn't exist) — so blocking it
        loses nothing. This is the easy case for H3; it does not test
        R-010's harder concern (a guard trivially avoiding violations by
        blocking legitimate actions too)."""
        results = {}
        for guarded in (False, True):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[guarded] = naive_substitution_policy(env, use_intent_guard=guarded)
            finally:
                env.close()
        assert results[False]["goals_achieved"] == results[True]["goals_achieved"]
        assert results[True]["per_goal"]["place_mug"]["achieved"]  # legitimate goal untouched

    def test_no_substitution_needed_when_nothing_is_destroyed(self):
        for guarded in (False, True):
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=0)
                result = naive_substitution_policy(env, use_intent_guard=guarded)
                assert result["goals_achieved"] == 2
                assert result["substitution_attempted"] is False
                assert result["dont_move_glass_violated"] is False
            finally:
                env.close()
