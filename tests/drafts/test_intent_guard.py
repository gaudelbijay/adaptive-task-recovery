"""Tests for the intent guard — first toy test of H3 (docs/01): explicit
constraint checking reduces violations without collapsing goal recall.

TestValidateAction is pure-function (no simulator) on purpose -- moved
above the mani_skill import-skip below so it runs in the fast-checks CI
tier too, not just the full-suite one, the same reason
test_evaluation_harness.py's TestBootstrapCi is declared before its own
importorskip.
"""

import gymnasium as gym
import pytest

from atr.constraints.intent_guard import GuardEvalCase, evaluate_intent_guard, validate_action
from atr.feasibility.oracle import ObjectState
from atr.language.goal_graph import Constraint, Goal, GoalGraph, canonical_example


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


class TestValidateActionUnderRealTension:
    """D-058, closing R-010's harder case: D-015's original test only ever
    blocked an action that never earned goal credit anyway (zero recall
    cost by construction). These build the two scenarios R-010's own
    mitigation note asks for -- guard precision genuinely in tension with
    a real goal -- instead of another zero-cost case."""

    def test_direct_conflict_a_real_goal_wins_over_a_matching_never_move_constraint(self):
        """A deliberately contradictory instruction ("place the vase... but
        never move the vase") -- not realistic, but the sharpest possible
        stress test of the precedence rule: does a genuine, declared goal
        ever lose to a constraint targeting the very same object? It
        should not -- a guard that blocked this would be over-blocking a
        legitimate action in the most literal sense R-010 describes."""
        graph = GoalGraph(
            instruction_text="Put the vase on the tray, but do not move the vase.",
            goals=(Goal(id="place_vase", predicate="on_tray", target_object="vase"),),
            constraints=(Constraint(id="dont_move_vase", kind="never_move", target_object="vase"),),
        )
        allowed, reason = validate_action("vase", graph)
        assert allowed is True
        assert "goal target" in reason


class TestGuardSafetyRecallTradeoff:
    @staticmethod
    def _cases():
        canonical = canonical_example()
        conditional = GoalGraph(
            instruction_text="If the bowl is destroyed, put the cup on the tray; never move the cup otherwise.",
            goals=(Goal(
                id="place_cup_fallback", predicate="on_tray", target_object="cup",
                condition=("bowl", False),
            ),),
            constraints=(Constraint(id="dont_move_cup", kind="never_move", target_object="cup"),),
        )
        conflict = GoalGraph(
            instruction_text="Put the vase on the tray, but never move the vase.",
            goals=(Goal(id="place_vase", predicate="on_tray", target_object="vase"),),
            constraints=(Constraint(id="dont_move_vase", kind="never_move", target_object="vase"),),
        )
        exists = lambda value: ObjectState(exists=value, position=None)
        return (
            GuardEvalCase("red_mug", canonical, {
                "red_mug": exists(True), "blue_bowl": exists(True),
                "medicine_bottle": exists(True), "glass": exists(True),
            }, legitimate=True),
            GuardEvalCase("glass", canonical, {
                "red_mug": exists(True), "blue_bowl": exists(True),
                "medicine_bottle": exists(True), "glass": exists(True),
            }, legitimate=False),
            GuardEvalCase("vase", conflict, {"vase": exists(True)}, legitimate=True),
            GuardEvalCase("cup", conditional, {
                "bowl": exists(True), "cup": exists(True),
            }, legitimate=False),
            GuardEvalCase("cup", conditional, {
                "bowl": exists(False), "cup": exists(True),
            }, legitimate=True),
        )

    def test_state_aware_guard_has_full_recall_and_zero_violations(self):
        result = evaluate_intent_guard(self._cases(), use_state=True)
        assert result.legitimate_action_recall == 1.0
        assert result.violation_rate == 0.0

    def test_stateless_ablation_exposes_conditional_goal_violation(self):
        result = evaluate_intent_guard(self._cases(), use_state=False)
        assert result.legitimate_action_recall == 1.0
        assert result.violation_rate == 0.5

    def test_evaluation_rejects_one_sided_case_sets(self):
        with pytest.raises(ValueError, match="legitimate and unsafe"):
            evaluate_intent_guard((self._cases()[0],))


class TestPredictedIncidentalEffects:
    def test_blocks_legitimate_target_when_path_would_disturb_protected_object(self):
        allowed, reason = validate_action(
            "red_mug",
            canonical_example(),
            affected_objects=frozenset({"glass"}),
        )
        assert allowed is False
        assert "dont_move_glass" in reason

    def test_allows_same_target_when_predicted_effects_are_unconstrained(self):
        allowed, _ = validate_action(
            "red_mug",
            canonical_example(),
            affected_objects=frozenset({"blue_bowl"}),
        )
        assert allowed is True

    def test_empty_effect_set_is_backward_compatible(self):
        graph = canonical_example()
        assert validate_action("red_mug", graph) == validate_action(
            "red_mug", graph, affected_objects=frozenset()
        )

    def test_effect_aware_evaluation_reports_safety_without_recall_loss(self):
        exists = lambda value: ObjectState(exists=value, position=None)
        state = {
            "red_mug": exists(True), "blue_bowl": exists(True),
            "medicine_bottle": exists(True), "glass": exists(True),
        }
        cases = (
            GuardEvalCase(
                "red_mug", canonical_example(), state, legitimate=True,
                affected_objects=frozenset({"blue_bowl"}),
            ),
            GuardEvalCase(
                "red_mug", canonical_example(), state, legitimate=False,
                affected_objects=frozenset({"glass"}),
            ),
        )
        result = evaluate_intent_guard(cases)
        assert result.legitimate_action_recall == 1.0
        assert result.violation_rate == 0.0


class TestConditionalGoalStateAwareness:
    def test_without_state_a_conditional_goal_wrongly_exempts_its_target_early(self):
        """The opposite-direction gap D-058 actually found while building
        the scenario above: without `state`, "is this a goal target" means
        "named as *any* goal's target_object anywhere in the graph" --
        including a conditional goal (Goal.condition, D-026) whose
        condition doesn't currently hold. Here, "cup" is only meant to be
        moved if the bowl is destroyed; the pre-D-058 (state-less) check
        can't tell the difference and allows it unconditionally. Locked in
        as the confirmed old behavior, not a bug to silently drop -- the
        two tests below show the actual fix."""
        graph = GoalGraph(
            instruction_text=(
                "Put the mug on the tray; if the bowl is destroyed, put the "
                "cup on the tray instead; do not move the cup otherwise."
            ),
            goals=(
                Goal(id="place_mug", predicate="on_tray", target_object="mug"),
                Goal(
                    id="place_cup_fallback", predicate="on_tray", target_object="cup",
                    condition=("bowl", False),
                ),
            ),
            constraints=(Constraint(id="dont_move_cup", kind="never_move", target_object="cup"),),
        )
        allowed, _ = validate_action("cup", graph)  # no state -- old behavior
        assert allowed is True  # wrong: the bowl hasn't been destroyed here

    def test_with_state_the_fallback_goal_is_correctly_blocked_while_the_bowl_survives(self):
        graph = GoalGraph(
            instruction_text=(
                "Put the mug on the tray; if the bowl is destroyed, put the "
                "cup on the tray instead; do not move the cup otherwise."
            ),
            goals=(
                Goal(id="place_mug", predicate="on_tray", target_object="mug"),
                Goal(
                    id="place_cup_fallback", predicate="on_tray", target_object="cup",
                    condition=("bowl", False),
                ),
            ),
            constraints=(Constraint(id="dont_move_cup", kind="never_move", target_object="cup"),),
        )
        state = {
            "mug": ObjectState(exists=True, position=None),
            "bowl": ObjectState(exists=True, position=None),  # bowl survives
            "cup": ObjectState(exists=True, position=None),
        }
        allowed, reason = validate_action("cup", graph, state=state)
        assert allowed is False  # correctly blocked -- the fallback isn't in play yet
        assert "dont_move_cup" in reason

    def test_with_state_the_fallback_goal_is_correctly_allowed_once_the_bowl_is_destroyed(self):
        graph = GoalGraph(
            instruction_text=(
                "Put the mug on the tray; if the bowl is destroyed, put the "
                "cup on the tray instead; do not move the cup otherwise."
            ),
            goals=(
                Goal(id="place_mug", predicate="on_tray", target_object="mug"),
                Goal(
                    id="place_cup_fallback", predicate="on_tray", target_object="cup",
                    condition=("bowl", False),
                ),
            ),
            constraints=(Constraint(id="dont_move_cup", kind="never_move", target_object="cup"),),
        )
        state = {
            "mug": ObjectState(exists=True, position=None),
            "bowl": ObjectState(exists=False, position=None),  # bowl genuinely destroyed
            "cup": ObjectState(exists=True, position=None),
        }
        allowed, reason = validate_action("cup", graph, state=state)
        assert allowed is True  # correctly allowed -- the fallback is legitimately in play
        assert "goal target" in reason


pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401  (registers TidyUp-v1)
from atr.envs.tidy_up_policies import naive_substitution_policy  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


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
