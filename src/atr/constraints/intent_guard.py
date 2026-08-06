"""Intent guard — first toy test of H3 (docs/01): "explicit goal/constraint
checking reduces semantic and constraint violations with an acceptable
trade-off in achievable-goal recall."

docs/07's intent model calls for distinguishing "an explicitly allowed
substitute from a semantically convenient but unauthorized replacement."
This is that check, in its smallest possible form: reject a candidate
action if it targets an object under a hard `never_move` constraint that
no actual goal in the graph requires touching.

R-010 (ai-notes/issues_and_risks.md) flags that D-015's original test
only ever exercised the easy case (blocking an action that never earned
goal credit anyway, so recall cost was zero by construction) and never
tested guard precision under real tension with a legitimate goal. D-058
built that harder case directly: a goal in direct target conflict with a
never_move constraint on the same object confirms the precedence rule
below is sound (goal wins, not over-blocked) -- see
tests/drafts/test_intent_guard.py's TestValidateAction. But it also found
a real, opposite-direction gap: without `state`, "is this a goal target"
means "named as *any* goal's target_object anywhere in the graph,"
including a conditional goal (`Goal.condition`, D-026) whose condition
doesn't currently hold -- confirmed too *permissive* in that case, which
is a genuine tension the original easy-case test couldn't have surfaced
either. See D-058.
"""

from __future__ import annotations

from atr.feasibility.oracle import WorldState, goal_feasible
from atr.language.goal_graph import GoalGraph


def validate_action(
    target_object: str, graph: GoalGraph, state: WorldState | None = None,
) -> tuple[bool, str]:
    """(allowed, reason). Rejects only objects under a never_move constraint
    that aren't themselves a goal target — i.e., moving them could never be
    legitimate, not just currently inconvenient.

    `state` (optional, D-058): without it, "goal target" means "named as
    *any* goal's target_object anywhere in the graph" — including a
    conditional goal (`Goal.condition`) whose condition doesn't currently
    hold, which is too permissive (a fallback goal that isn't actually in
    play yet shouldn't exempt its target object from a real constraint).
    Pass `state` — the same privileged world state
    `feasibility_aware_policy`/`naive_substitution_policy` already read —
    to check `goal_feasible()` instead, which resolves `Goal.condition`
    correctly. Kept optional rather than required: `validate_action()`
    predates D-026's conditional goals, and no goal in this project has
    ever combined `condition` with a `never_move` constraint on its own
    target object before D-058's test, so this is the minimal fix, not a
    signature break for callers that have no state to give it."""
    if state is not None:
        is_goal_target = any(
            g.target_object == target_object and goal_feasible(g, state) for g in graph.goals
        )
    else:
        is_goal_target = any(g.target_object == target_object for g in graph.goals)
    if is_goal_target:
        return True, "ok: object is an actual goal target"
    for constraint in graph.constraints:
        if constraint.kind == "never_move" and constraint.target_object == target_object:
            return False, f"blocked: would violate constraint '{constraint.id}' (never_move on {target_object})"
    return True, "ok"
