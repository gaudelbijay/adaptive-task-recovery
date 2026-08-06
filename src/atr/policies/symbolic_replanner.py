"""Symbolic replanner with learned state (D-067) -- docs/10's required
baseline, distinct from every existing policy in this project: all of
them (`baselines.py`'s static/oracle-feasibility/naive-substitution,
`q_learning.py`, `imitation.py`, `domain_randomized.py`) make one fixed
pass through `graph.goals` in tuple order, checking each goal in place.
None of them actually *searches* over alternative orderings -- `Goal.
priority`/`Goal.depends_on` exist in the schema (D-013/D-037) but
nothing before this used them to choose a plan, only to gate a fixed
order (`goal_dependencies_satisfied()`, read sequentially).

`plan()` below is a real (if tiny) planner: given a `GoalGraph` and a
feasibility estimate -- "learned state" means this can be privileged
oracle state OR a perceptual judgment (CLIP/DINOv2), the function itself
doesn't care which, it just wants an `exists: dict[str, bool]` -- it
enumerates every ordering of the not-yet-achieved goals, keeps only the
orderings where each goal's `depends_on` is satisfied by goals earlier
in that same ordering, scores each valid ordering by the total
`priority + 1` of the goals it can actually achieve (feasible and
dependency-satisfied), and returns the highest-scoring one.
`dependent_goals_example()` (`atr.language.goal_graph`) is the real test
case this matters for: `place_bowl` (priority 1) depends on `place_mug`
(priority 0) being *achieved*, not just feasible -- a genuine planning
decision (attempt the lower-priority prerequisite first, to unlock the
higher-value goal) that a fixed-order walk gets right only by
coincidence of tuple order, not by reasoning about it.

`run_replanner_episode()` genuinely *replans*, not just plans once: it
calls `plan()` again after every single goal attempt, given whatever
actually happened (an attempt can still fail even when perceived
feasible), rather than committing blindly to the rest of an earlier
plan that might now be stale.
"""

from __future__ import annotations

import itertools
from typing import Callable

from atr.feasibility.oracle import ObjectState, WorldState, goal_dependencies_satisfied, goal_feasible
from atr.language.goal_graph import Goal, GoalGraph
from atr.policies.baselines import _summarize

ExistsFn = Callable[..., dict[str, bool]]


def _state_from_exists(exists: dict[str, bool]) -> WorldState:
    """Builds the minimal `WorldState` `goal_feasible()` needs (existence
    only -- position/up_vector aren't read by feasibility checks) from a
    plain `{object_id: exists}` dict, whether that dict came from
    privileged state or a perceptual model's judgment."""
    return {obj: ObjectState(exists=value, position=None) for obj, value in exists.items()}


def plan(
    graph: GoalGraph, exists: dict[str, bool], achieved_ids: frozenset[str] = frozenset(),
) -> list[Goal]:
    """Returns the highest-scoring valid ordering of not-yet-achieved
    goals, given a feasibility estimate and which goals are already
    achieved. "Valid" means every included goal's `depends_on` is
    satisfied by goals earlier in the *same* ordering (or already in
    `achieved_ids`) and the goal is feasible under `exists`
    (`goal_feasible()`, which also resolves `Goal.condition`, D-026).
    Score is `sum(goal.priority + 1)` over included goals -- `+ 1` so a
    priority-0 goal (every goal in this project except
    `dependent_goals_example()`'s `place_bowl`) still contributes,
    keeping "achieve more goals" the default objective with `priority`
    as a real tiebreaker/weight, not the only thing that matters."""
    state = _state_from_exists(exists)
    remaining = [g for g in graph.goals if g.id not in achieved_ids]

    best_plan: tuple[Goal, ...] = ()
    best_score = -1
    for ordering in itertools.permutations(remaining):
        valid: list[Goal] = []
        satisfied = set(achieved_ids)
        score = 0
        for goal in ordering:
            if not goal_dependencies_satisfied(goal, satisfied):
                continue
            if not goal_feasible(goal, state):
                continue
            valid.append(goal)
            satisfied.add(goal.id)  # assumed achieved, for planning purposes only
            score += goal.priority + 1
        if score > best_score:
            best_score = score
            best_plan = tuple(valid)
    return list(best_plan)


def run_replanner_episode(
    env, graph: GoalGraph, attempt_goal_fn: Callable, tray_slots: list, exists_fn: ExistsFn,
) -> dict:
    """Drives a full episode by replanning after every goal attempt --
    `exists_fn(env) -> {object_id: exists}` is called fresh each time
    (privileged `env.unwrapped._exists`/`_world_state()`-derived, or a
    real perceptual read, e.g. `visual_object_exists()` per calibrated
    object -- this function doesn't know or care which). If `plan()`
    returns nothing plannable right now (nothing left is both feasible
    and dependency-satisfied), the highest-priority remaining goal is
    explicitly recorded as skipped -- never silently dropped."""
    tray_slot_by_id = {goal.id: tray_slots[i] for i, goal in enumerate(graph.goals)}
    achieved_ids: set[str] = set()
    per_goal: dict[str, dict] = {}

    while len(per_goal) < len(graph.goals):
        exists = exists_fn(env)
        current_plan = plan(graph, exists, frozenset(achieved_ids))
        not_yet_attempted = [g for g in graph.goals if g.id not in per_goal]
        plannable_ids = {g.id for g in current_plan}
        next_goal = current_plan[0] if current_plan and current_plan[0].id in plannable_ids else None

        if next_goal is not None:
            result = attempt_goal_fn(env, next_goal, tray_slot_by_id[next_goal.id])
            per_goal[next_goal.id] = result
            if result["achieved"]:
                achieved_ids.add(next_goal.id)
        else:
            goal = max(not_yet_attempted, key=lambda g: g.priority)
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}

    return _summarize(per_goal)
