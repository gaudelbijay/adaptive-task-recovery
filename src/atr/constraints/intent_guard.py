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

D-082 aggregates those cases instead of leaving them as isolated assertions:
`evaluate_intent_guard()` reports legitimate-action recall and unsafe-action
violation rate separately. On the constructible evaluation set, state-aware
validation achieves recall 1.0 and violation rate 0.0; the stateless ablation
keeps recall 1.0 but permits half of unsafe candidates.

D-083 makes R-010's remaining side-effect case representable at the semantic
skill boundary: callers may pass `affected_objects` predicted by a motion or
skill model, and the guard checks every predicted effect, not only the named
target. It does not itself predict contacts or trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass

from atr.feasibility.oracle import WorldState, goal_feasible
from atr.language.goal_graph import GoalGraph


@dataclass(frozen=True)
class GuardEvalCase:
    """One candidate action with an independently declared safety label."""

    target_object: str
    graph: GoalGraph
    state: WorldState
    legitimate: bool
    affected_objects: frozenset[str] = frozenset()


@dataclass(frozen=True)
class GuardEvaluation:
    legitimate_action_recall: float
    violation_rate: float
    allowed: tuple[bool, ...]


def evaluate_intent_guard(
    cases: tuple[GuardEvalCase, ...], *, use_state: bool = True,
) -> GuardEvaluation:
    """Measure H3's safety/recall trade-off without collapsing either side.

    `legitimate_action_recall` is the fraction of independently labelled
    legitimate candidates the guard permits. `violation_rate` is the fraction
    of unsafe candidates it permits. Both classes must be present so a guard
    cannot look safe by doing nothing or look permissive by seeing no hazards.
    """
    legitimate = [case for case in cases if case.legitimate]
    unsafe = [case for case in cases if not case.legitimate]
    if not legitimate or not unsafe:
        raise ValueError("guard evaluation requires legitimate and unsafe cases")
    allowed = tuple(
        validate_action(
            case.target_object, case.graph, state=case.state if use_state else None,
            affected_objects=case.affected_objects,
        )[0]
        for case in cases
    )
    legitimate_allowed = sum(
        decision for decision, case in zip(allowed, cases) if case.legitimate
    )
    unsafe_allowed = sum(
        decision for decision, case in zip(allowed, cases) if not case.legitimate
    )
    return GuardEvaluation(
        legitimate_action_recall=legitimate_allowed / len(legitimate),
        violation_rate=unsafe_allowed / len(unsafe),
        allowed=allowed,
    )


def validate_action(
    target_object: str,
    graph: GoalGraph,
    state: WorldState | None = None,
    affected_objects: frozenset[str] | None = None,
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
    # D-083: the target is always an effect; callers with a motion/skill
    # predictor may additionally name objects the trajectory could disturb.
    effects = frozenset({target_object}) | (affected_objects or frozenset())
    for affected in effects:
        if state is not None:
            is_goal_target = any(
                g.target_object == affected and goal_feasible(g, state)
                for g in graph.goals
            )
        else:
            is_goal_target = any(g.target_object == affected for g in graph.goals)
        if is_goal_target:
            continue
        for constraint in graph.constraints:
            if constraint.kind == "never_move" and constraint.target_object == affected:
                return False, (
                    f"blocked: would violate constraint '{constraint.id}' "
                    f"(never_move on {affected})"
                )
    if target_object in effects and any(
        g.target_object == target_object
        and (state is None or goal_feasible(g, state))
        for g in graph.goals
    ):
        return True, "ok: object is an actual goal target"
    return True, "ok"
