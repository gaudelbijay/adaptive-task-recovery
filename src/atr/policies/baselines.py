"""Env-agnostic policy-decision logic (D-040), promoted from four
near-identical spikes/task_schema_draft/policy_baselines*.py copies (one
per embodiment: panda tabletop, G1 humanoid, ReplicaCAD+Fetch,
G1-in-ReplicaCAD). Same parameterization pattern rl_policy.py's
train_q_table() already used for the same reason (D-030): the algorithm
was identical across copies, only `attempt_goal`/tray geometry/example
graph differed.

Not hypothetical: this duplication already caused a real bug. D-037 added
`goal_dependencies_satisfied()` gating to `feasibility_aware_policy()` --
but only in `policy_baselines.py`, since that was the only copy touched.
The other three variants silently kept the old, ungated logic until this
promotion folded all four into one implementation. Exactly the "same fix
landing once, not everywhere" risk D-030's own docstring already warned
about for a different pair of duplicates.

Each spike env file keeps its own `attempt_goal(env, goal, tray_slot_xyz,
...) -> {"achieved": bool, "steps_used": int, "skipped": bool}` --
the one thing that's genuinely embodiment/env-specific (Cartesian IK,
joint-space reach, or navigate-then-reach) -- and its own tray geometry,
then calls into these functions with both as parameters.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from atr.constraints.intent_guard import validate_action
from atr.feasibility.oracle import (
    constraint_violated,
    goal_achieved,
    goal_dependencies_satisfied,
    goal_feasible,
)
from atr.language.goal_graph import Goal, GoalGraph

AttemptGoalFn = Callable[..., dict]


def _summarize(per_goal: dict) -> dict:
    return {
        "per_goal": per_goal,
        "goals_achieved": sum(r["achieved"] for r in per_goal.values()),
        "total_steps": sum(r["steps_used"] for r in per_goal.values()),
        "wasted_steps": sum(
            r["steps_used"] for r in per_goal.values() if not r["achieved"] and not r["skipped"]
        ),
        # D-094: zero for embodiments without D-093 navigation metadata,
        # directly measurable for ReplicaCAD Fetch without special-casing
        # policy names in the evaluation harness.
        "navigation_replans": sum(
            bool(r.get("navigation_replanned", False)) for r in per_goal.values()
        ),
        "navigation_safety_blocks": sum(
            bool(r.get("navigation_safety_screened", False))
            and bool(r.get("blocked_reason"))
            and bool(r.get("skipped", False))
            for r in per_goal.values()
        ),
        # D-110: execution/planning failures are distinct from semantic
        # safety blocks. Keep the aggregate generic so later controller
        # failure reasons use the same observable contract.
        "navigation_failures": sum(
            bool(r.get("navigation_failure_reason")) for r in per_goal.values()
        ),
    }


def static_policy(
    env, graph: GoalGraph, attempt_goal_fn: AttemptGoalFn, tray_slots: list[np.ndarray],
) -> dict:
    """Attempts every goal in order, regardless of feasibility."""
    per_goal = {
        goal.id: attempt_goal_fn(env, goal, tray_slots[i]) for i, goal in enumerate(graph.goals)
    }
    return _summarize(per_goal)


def feasibility_aware_policy(
    env, graph: GoalGraph, attempt_goal_fn: AttemptGoalFn, tray_slots: list[np.ndarray],
) -> dict:
    """Checks goal_feasible() and goal_dependencies_satisfied() (D-037,
    universal here since this promotion -- see module docstring) before
    committing to the physical attempt; skips a goal immediately if
    either fails."""
    per_goal = {}
    achieved_ids: set[str] = set()
    for i, goal in enumerate(graph.goals):
        state = env.unwrapped._world_state()
        if not goal_feasible(goal, state) or not goal_dependencies_satisfied(goal, achieved_ids):
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
            continue
        result = attempt_goal_fn(env, goal, tray_slots[i])
        per_goal[goal.id] = result
        if result["achieved"]:
            achieved_ids.add(goal.id)
    return _summarize(per_goal)


def naive_substitution_policy(
    env, graph: GoalGraph, attempt_goal_fn: AttemptGoalFn, tray_slots: list[np.ndarray],
    use_intent_guard: bool = False,
    settle_steps: int = 0,
    settle_action: np.ndarray | None = None,
    predict_goal_effects_fn: Callable | None = None,
) -> dict:
    """The "invalid agent" from docs/01's own worked example: rather than
    accepting an infeasible goal, substitutes an unrequested object --
    whichever one carries this graph's `never_move` constraint -- to fill
    the tray slot instead. Generalized from four copies that each
    hardcoded a different literal object name ("glass"/"master_chef_can"/
    "bowl") for exactly this same role; deriving it from the graph's own
    `never_move` constraint instead reproduces every one of those
    hardcoded results exactly, since each graph only ever has one.

    `settle_steps`/`settle_action`: three of the four original copies
    stepped a neutral action a few times before capturing `initial_state`
    -- spawned objects still settling onto a real surface would otherwise
    register as a false never-move violation before anything touched
    them. The panda/tabletop variant never needed this (default
    `settle_steps=0` reproduces its behavior exactly)."""
    if settle_steps:
        for _ in range(settle_steps):
            env.step(settle_action)
    initial_state = env.unwrapped._world_state()
    per_goal = {}
    substitution_attempted = False

    guarded_constraint = next(c for c in graph.constraints if c.kind == "never_move")
    substitute_object = guarded_constraint.target_object

    for i, goal in enumerate(graph.goals):
        state = env.unwrapped._world_state()
        if goal_feasible(goal, state):
            if use_intent_guard and predict_goal_effects_fn is not None:
                affected_objects = frozenset(predict_goal_effects_fn(env, goal))
                allowed, reason = validate_action(
                    goal.target_object, graph, state=state,
                    affected_objects=affected_objects,
                )
                if not allowed:
                    per_goal[goal.id] = {
                        "achieved": False, "steps_used": 0, "skipped": True,
                        "substitution_attempted": False,
                        "blocked_reason": reason,
                        "predicted_affected_objects": sorted(affected_objects),
                    }
                    continue
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
            continue

        if use_intent_guard:
            # D-058: pass `state` so a conditional goal (Goal.condition)
            # only exempts its target object when actually in play, not
            # merely because it's declared somewhere in the graph.
            allowed, reason = validate_action(substitute_object, graph, state=state)
        else:
            allowed, reason = True, "unchecked (no intent guard)"

        if not allowed:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "substitution_attempted": False, "blocked_reason": reason,
            }
            continue

        substitution_attempted = True
        fake_goal = Goal(
            id=f"substitute_for_{goal.id}", predicate="on_tray", target_object=substitute_object,
        )
        substitution_result = attempt_goal_fn(env, fake_goal, tray_slots[i])
        per_goal[goal.id] = {
            # Never credited: moving the substitute onto the tray does not
            # satisfy the real (infeasible) goal, no matter how the
            # substitution attempt itself goes.
            "achieved": False,
            "steps_used": substitution_result["steps_used"],
            "skipped": False,
            "substitution_attempted": True,
        }

    final_state = env.unwrapped._world_state()
    result = _summarize(per_goal)
    result[f"dont_move_{substitute_object}_violated"] = constraint_violated(
        guarded_constraint, initial_state, final_state
    )
    result["substitution_attempted"] = substitution_attempted
    return result
