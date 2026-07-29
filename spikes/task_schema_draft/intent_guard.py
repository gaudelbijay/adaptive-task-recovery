"""Intent guard — first toy test of H3 (docs/01): "explicit goal/constraint
checking reduces semantic and constraint violations with an acceptable
trade-off in achievable-goal recall."

docs/07's intent model calls for distinguishing "an explicitly allowed
substitute from a semantically convenient but unauthorized replacement."
This is that check, in its smallest possible form: reject a candidate
action if it targets an object under a hard `never_move` constraint that
no actual goal in the graph requires touching.
"""

from __future__ import annotations

from task_schema_draft.goal_graph import GoalGraph


def validate_action(target_object: str, graph: GoalGraph) -> tuple[bool, str]:
    """(allowed, reason). Rejects only objects under a never_move constraint
    that aren't themselves a goal target — i.e., moving them could never be
    legitimate, not just currently inconvenient."""
    is_goal_target = any(g.target_object == target_object for g in graph.goals)
    if is_goal_target:
        return True, "ok: object is an actual goal target"
    for constraint in graph.constraints:
        if constraint.kind == "never_move" and constraint.target_object == target_object:
            return False, f"blocked: would violate constraint '{constraint.id}' (never_move on {target_object})"
    return True, "ok"
