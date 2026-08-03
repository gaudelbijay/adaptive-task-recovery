"""Pure-function tests for the goal graph + oracle feasibility draft.

No simulator required — these test the schema logic in isolation, per
docs/04-benchmark-environment.md's "validate oracle labels on hand-authored
cases" guidance.
"""

import numpy as np

from atr.language.goal_graph import Goal, canonical_example, dependent_goals_example
from atr.feasibility.oracle import (
    ObjectState,
    evaluate_goal_graph,
    goal_achieved,
    goal_dependencies_satisfied,
    goal_feasible,
)


def _state(**objects: tuple[bool, tuple[float, float, float] | None, float | None]) -> dict:
    """objects: name -> (exists, position, up_z)"""
    out = {}
    for name, (exists, pos, up_z) in objects.items():
        out[name] = ObjectState(
            exists=exists,
            position=None if pos is None else np.array(pos),
            up_vector=None if up_z is None else np.array([0.0, 0.0, up_z]),
        )
    return out


def _nominal_state():
    return _state(
        red_mug=(True, (0.15, -0.15, 0.04), None),
        blue_bowl=(True, (0.15, 0.15, 0.025), None),
        tray=(True, (0.4, 0.0, 0.005), None),
        medicine_bottle=(True, (0.0, -0.2, 0.05), 1.0),
        glass=(True, (0.0, 0.2, 0.045), None),
    )


class TestCanonicalExample:
    def test_matches_docs_01_worked_example(self):
        graph = canonical_example()
        assert {g.target_object for g in graph.goals} == {"red_mug", "blue_bowl"}
        assert {c.target_object for c in graph.constraints} == {"medicine_bottle", "glass"}

    def test_nominal_state_is_fully_feasible_and_unviolated(self):
        graph = canonical_example()
        state = _nominal_state()
        result = evaluate_goal_graph(graph, initial_state=state, current_state=state)
        assert all(result["goal_feasibility"].values())
        assert not any(result["constraint_violations"].values())


class TestBowlDestroyedIntervention:
    def test_bowl_goal_infeasible_mug_goal_unaffected(self):
        """The exact scenario from docs/01: bowl breaks -> bowl goal
        infeasible, mug goal still feasible, no constraint violated."""
        graph = canonical_example()
        initial = _nominal_state()
        after = dict(initial)
        after["blue_bowl"] = ObjectState(exists=False, position=None, up_vector=None)

        result = evaluate_goal_graph(graph, initial_state=initial, current_state=after)
        assert result["goal_feasibility"] == {"place_mug": True, "place_bowl": False}
        assert not any(result["constraint_violations"].values())


class TestConditionalGoal:
    """PROPOSED extension (D-026, not yet reviewed): Goal.condition gates
    whether a goal is "in play" at all, independent of its own target
    object -- e.g. a fallback goal that only matters once the thing it's
    a fallback *for* is gone."""

    _fallback = Goal(
        id="place_backup_bowl", predicate="on_tray", target_object="backup_bowl",
        condition=("blue_bowl", False),  # only in play once blue_bowl is gone
    )

    def test_fallback_goal_infeasible_while_trigger_object_exists(self):
        state = _nominal_state()
        state.update(_state(backup_bowl=(True, (0.2, 0.2, 0.03), None)))
        assert goal_feasible(self._fallback, state) is False  # blue_bowl still exists -- condition not met

    def test_fallback_goal_feasible_once_trigger_object_destroyed(self):
        state = _nominal_state()
        state.update(_state(blue_bowl=(False, None, None), backup_bowl=(True, (0.2, 0.2, 0.03), None)))
        assert goal_feasible(self._fallback, state) is True

    def test_fallback_goal_still_infeasible_if_its_own_object_also_gone(self):
        state = _nominal_state()
        state.update(_state(blue_bowl=(False, None, None), backup_bowl=(False, None, None)))
        assert goal_feasible(self._fallback, state) is False  # condition met, but nothing to place either


class TestGoalDependency:
    """D-037, resolving the D-013 review's open question 3: Goal.depends_on
    was defined from the start but never read by any function. A hard
    prerequisite gate, distinct from Goal.priority (nothing reads it) and
    Goal.condition (gates on object existence, not another goal's
    completion) -- see goal_dependencies_satisfied()'s own docstring."""

    def test_dependent_goals_example_matches_its_own_description(self):
        graph = dependent_goals_example()
        mug, bowl = graph.goals
        assert mug.depends_on == ()
        assert bowl.depends_on == ("place_mug",)

    def test_goal_with_no_dependencies_is_always_satisfied(self):
        mug = dependent_goals_example().goals[0]
        assert goal_dependencies_satisfied(mug, achieved_goal_ids=set()) is True

    def test_goal_blocked_until_its_dependency_is_achieved(self):
        bowl = dependent_goals_example().goals[1]
        assert goal_dependencies_satisfied(bowl, achieved_goal_ids=set()) is False
        assert goal_dependencies_satisfied(bowl, achieved_goal_ids={"place_mug"}) is True

    def test_unrelated_achieved_ids_do_not_satisfy_a_real_dependency(self):
        bowl = dependent_goals_example().goals[1]
        assert goal_dependencies_satisfied(bowl, achieved_goal_ids={"some_other_goal"}) is False


class TestConstraintViolations:
    def test_moving_glass_violates_constraint(self):
        graph = canonical_example()
        initial = _nominal_state()
        after = dict(initial)
        after["glass"] = ObjectState(exists=True, position=np.array([0.1, 0.2, 0.045]), up_vector=None)

        result = evaluate_goal_graph(graph, initial_state=initial, current_state=after)
        assert result["constraint_violations"]["dont_move_glass"] is True
        assert result["constraint_violations"]["keep_medicine_upright"] is False

    def test_tipping_medicine_violates_constraint(self):
        graph = canonical_example()
        initial = _nominal_state()
        after = dict(initial)
        after["medicine_bottle"] = ObjectState(
            exists=True, position=np.array([0.0, -0.2, 0.05]), up_vector=np.array([1.0, 0.0, 0.0])
        )

        result = evaluate_goal_graph(graph, initial_state=initial, current_state=after)
        assert result["constraint_violations"]["keep_medicine_upright"] is True
        assert result["constraint_violations"]["dont_move_glass"] is False

    def test_destroyed_object_does_not_count_as_a_violation(self):
        """A constraint on an object that no longer exists is a feasibility
        question (goal_feasible), not a constraint-violation question —
        see oracle_feasibility.py's docstring on this distinction."""
        graph = canonical_example()
        initial = _nominal_state()
        after = dict(initial)
        after["glass"] = ObjectState(exists=False, position=None, up_vector=None)

        result = evaluate_goal_graph(graph, initial_state=initial, current_state=after)
        assert result["constraint_violations"]["dont_move_glass"] is False


class TestGoalAchieved:
    TRAY_POSITION = np.array([0.4, 0.0, 0.005])
    TRAY_HALF_SIZES = (0.15, 0.2, 0.005)

    def test_object_on_tray_is_achieved(self):
        graph = canonical_example()
        state = _nominal_state()
        state["red_mug"] = ObjectState(exists=True, position=np.array([0.4, -0.08, 0.005]))
        assert goal_achieved(graph.goals[0], state, self.TRAY_POSITION, self.TRAY_HALF_SIZES)

    def test_object_off_tray_is_not_achieved(self):
        graph = canonical_example()
        state = _nominal_state()  # red_mug still at its spawn position, far from the tray
        assert not goal_achieved(graph.goals[0], state, self.TRAY_POSITION, self.TRAY_HALF_SIZES)

    def test_nonexistent_object_is_not_achieved(self):
        graph = canonical_example()
        state = _nominal_state()
        state["blue_bowl"] = ObjectState(exists=False, position=None)
        assert not goal_achieved(graph.goals[1], state, self.TRAY_POSITION, self.TRAY_HALF_SIZES)

    def test_tolerates_float_precision_at_exact_tray_height(self):
        """Regression test: a real teleport-onto-tray in tidy_up_env
        produced dz = -1.1e-10 (float32/float64 mixing) at an object sitting
        exactly at tray height, which a strict `0 <= dz` check rejected.
        See oracle_feasibility.py's goal_achieved for the -1e-4 tolerance fix."""
        graph = canonical_example()
        state = _nominal_state()
        position = self.TRAY_POSITION.copy()
        position[2] = np.float32(self.TRAY_POSITION[2])  # float32 round-trip, same as sim state
        state["red_mug"] = ObjectState(exists=True, position=position)
        assert goal_achieved(graph.goals[0], state, self.TRAY_POSITION, self.TRAY_HALF_SIZES)
