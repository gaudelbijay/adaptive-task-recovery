"""Pure-function tests for the goal graph + oracle feasibility draft.

No simulator required — these test the schema logic in isolation, per
docs/04-benchmark-environment.md's "validate oracle labels on hand-authored
cases" guidance.
"""

import numpy as np

from task_schema_draft.goal_graph import canonical_example
from task_schema_draft.oracle_feasibility import ObjectState, evaluate_goal_graph, goal_achieved


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
