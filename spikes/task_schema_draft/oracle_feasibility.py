"""Oracle feasibility and constraint-violation checking over privileged state.

Per docs/04-benchmark-environment.md "Oracle feasibility": "For small
domains, compute feasibility with exhaustive search... over privileged
state" — this domain (2 goals, 2 constraints, 5 objects) is exactly that
small case. Per "Humanoid validity requirements": "Do not label a goal
infeasible merely because one grasp or controller rollout fails" — these
functions check object *existence*, never attempted-motion outcomes, so a
failed grasp attempt never gets confused with a genuinely destroyed object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task_schema_draft.goal_graph import Constraint, Goal, GoalGraph


@dataclass
class ObjectState:
    """Privileged, existence-and-pose-only snapshot of one object."""

    exists: bool
    position: np.ndarray | None  # world-frame xyz, or None if it doesn't exist
    up_vector: np.ndarray | None = None  # object's local +z axis in world frame


WorldState = dict[str, ObjectState]


def goal_feasible(goal: Goal, state: WorldState) -> bool:
    """A goal is infeasible iff its target object no longer exists. Does not
    check reachability/graspability — that's a controller-level concern
    (docs/04), kept separate on purpose."""
    obj = state.get(goal.target_object)
    return obj is not None and obj.exists


def constraint_violated(constraint: Constraint, initial_state: WorldState, current_state: WorldState) -> bool:
    obj0 = initial_state.get(constraint.target_object)
    obj1 = current_state.get(constraint.target_object)
    if obj0 is None or obj1 is None or not obj1.exists:
        # Can't check position/orientation of something that no longer
        # exists — that's a feasibility question (goal_feasible), not a
        # constraint-violation question. Not violated by this check.
        return False
    if constraint.kind == "never_move":
        return bool(np.linalg.norm(obj1.position - obj0.position) > constraint.tolerance)
    if constraint.kind == "maintain_orientation":
        return bool(obj1.up_vector[2] < constraint.tolerance)
    raise ValueError(f"Unknown constraint kind: {constraint.kind}")


def evaluate_goal_graph(graph: GoalGraph, initial_state: WorldState, current_state: WorldState) -> dict:
    """The oracle predicates docs/04's task schema asks every episode to log:
    goal completion/feasibility and constraint violation, all off privileged
    state."""
    return {
        "goal_feasibility": {g.id: goal_feasible(g, current_state) for g in graph.goals},
        "constraint_violations": {
            c.id: constraint_violated(c, initial_state, current_state) for c in graph.constraints
        },
    }
