"""Promoted to src/atr/ 2026-08-02 (D-038) -- see ai-notes/decisions.md.
Originally stage 2 of docs/00-project-overview.md's build-up order: parse
an actual instruction sentence into a GoalGraph, instead of writing one
by hand.

Deliberately a controlled grammar, not open-ended NLU, per
docs/04-benchmark-environment.md: "Language templates should support
conjunction, ordering, exclusion, conditional goals, and preferences. Hold
out paraphrases and compositions..." Conjunction and exclusion were the
first two covered (every hand-authored GoalGraph in this directory already
used them). Ordering/priority and conditional goals are covered now too --
see "Ordering/priority" and "Conditional goals" below. Preferences (soft,
non-binding wishes as opposed to hard goals/constraints) remain
unimplemented -- no schema field exists for them yet, and adding one is a
schema decision, not just a grammar one; see D-026 in ai-notes/decisions.md.

Object names are resolved against a caller-supplied vocabulary (the set of
object ids that actually exist in a given scene), not guessed from open
vocabulary -- this project already assumes a closed, object-centric world
(docs/00 "Scope (v1)"), so the parser does too. An unrecognized clause
raises rather than being silently dropped: silently ignoring a "do not
move X" clause would be exactly the kind of intent violation this project
exists to catch.

## Ordering/priority

"First put the mug on the tray, then put the bowl on the tray" assigns
Goal.priority in order of appearance among clauses that carry an explicit
order marker (first/second/third/then/next/finally/after that) -- 0, 1,
2, .... A goal clause with no marker keeps priority=0 (unchanged from
before this was added, so every existing instruction_text still parses
identically).

## Conditional goals

"If the blue bowl is destroyed, put the backup bowl on the tray instead"
sets the resulting Goal's `condition` field (D-026, reviewed and accepted
as-is D-037) to (trigger_object_id, required_exists). Parsed as a
*separate* pass, before the generic comma-based clause
splitter runs -- not as another clause type dispatched through
_classify_clause() like the others. Real reason, found by testing: the
generic splitter breaks on any comma immediately before a recognized verb
("put"/"place"), which is exactly the shape of "if X is Y, put Z on the
tray" -- the comma before "put" would otherwise get treated as an ordinary
clause boundary, splitting the condition from the goal it's supposed to
gate and leaving neither half parseable. Extracting conditional clauses
first, then splitting whatever's left, avoids that entirely.
"""

from __future__ import annotations

import re
from typing import Iterable

from atr.language.goal_graph import Constraint, Goal, GoalGraph

_ORDER_MARKERS = ("first", "second", "third", "then", "next", "finally", "after that")
_ORDER_PATTERN = "|".join(_ORDER_MARKERS)

_CLAUSE_VERBS = rf"put|place|keep|do not|don't|dont|never|leave|if|{_ORDER_PATTERN}"

_CLAUSE_SEP = re.compile(
    rf"(?:,\s*(?:and\s+)?|\.\s*|;\s*|\s+and\s+)(?=(?:{_CLAUSE_VERBS})\b)",
    re.IGNORECASE,
)

_GOAL_RE = re.compile(
    rf"^(?:(?P<order>{_ORDER_PATTERN}),?\s+)?(?:put|place)\s+(?P<objects>.+?)\s+on\s+the\s+\w+$",
    re.IGNORECASE,
)
_ORIENT_RE = re.compile(r"^keep\s+(?:the\s+)?(.+?)\s+upright$", re.IGNORECASE)
_NEVER_MOVE_RE = re.compile(
    r"^(?:do not|don't|dont|never)\s+(?:move|touch)\s+(?:the\s+)?(.+?)$",
    re.IGNORECASE,
)
_LEAVE_ALONE_RE = re.compile(r"^leave\s+(?:the\s+)?(.+?)\s+alone$", re.IGNORECASE)

_CONDITION_NEGATIVE_STATES = {"destroyed", "broken", "missing", "gone", "unavailable"}
_CONDITION_POSITIVE_STATES = {"present", "available", "intact"}
_CONDITIONAL_RE = re.compile(
    r"(?:^|,\s*(?:and\s+)?|\.\s*|;\s*|\s+and\s+)"  # leading connector (or start), consumed -- see module docstring
    r"if\s+(?:the\s+)?(?P<trigger>.+?)\s+is\s+"
    rf"(?P<state>{'|'.join(_CONDITION_NEGATIVE_STATES | _CONDITION_POSITIVE_STATES)})"
    r",?\s*(?:then\s+)?(?:put|place)\s+(?P<objects>.+?)\s+on\s+the\s+\w+(?:\s+instead)?"
    r"(?=[,.]|$)",
    re.IGNORECASE,
)


def _clean_phrase(text: str) -> str:
    text = text.strip(" .,;")
    if text.lower().startswith("the "):
        text = text[4:]
    return re.sub(r"\s+", " ", text).strip().lower()


def _split_object_list(text: str) -> list[str]:
    text = re.sub(r"\s+and\s+", ",", text, flags=re.IGNORECASE)
    return [_clean_phrase(part) for part in text.split(",") if _clean_phrase(part)]


def _resolve_object(phrase: str, known_objects: Iterable[str]) -> str:
    words = set(phrase.split())
    exact, partial = [], []
    for obj_id in known_objects:
        obj_words = set(obj_id.split("_"))
        if words == obj_words:
            exact.append(obj_id)
        elif words <= obj_words or obj_words <= words:
            partial.append(obj_id)
    if exact:
        return exact[0]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise ValueError(f"ambiguous object phrase {phrase!r}: matches {partial}")
    raise ValueError(f"unrecognized object phrase {phrase!r}: not in known_objects")


def _classify_clause(clause: str) -> tuple[str, list[str], str | None]:
    if m := _GOAL_RE.match(clause):
        return "goal", _split_object_list(m.group("objects")), m.group("order")
    if m := _ORIENT_RE.match(clause):
        return "orient", [_clean_phrase(m.group(1))], None
    if m := _NEVER_MOVE_RE.match(clause):
        return "never_move", [_clean_phrase(m.group(1))], None
    if m := _LEAVE_ALONE_RE.match(clause):
        return "never_move", [_clean_phrase(m.group(1))], None
    raise ValueError(f"unrecognized clause: {clause!r}")


def _extract_conditional_goals(text: str, known_objects: list[str]) -> tuple[list[Goal], str]:
    """Pulls out "if X is Y, put Z on the tray instead" clauses before
    generic splitting runs -- see module docstring for why this can't be
    just another _classify_clause() branch. Returns (conditional goals,
    remaining text with those clauses removed)."""
    matches = list(_CONDITIONAL_RE.finditer(text))
    goals: list[Goal] = []
    for m in matches:
        trigger_id = _resolve_object(_clean_phrase(m.group("trigger")), known_objects)
        required_exists = m.group("state").lower() in _CONDITION_POSITIVE_STATES
        for phrase in _split_object_list(m.group("objects")):
            obj_id = _resolve_object(phrase, known_objects)
            goals.append(
                Goal(
                    id=f"place_{obj_id}_if_{trigger_id}_{'present' if required_exists else 'gone'}",
                    predicate="on_tray", target_object=obj_id,
                    condition=(trigger_id, required_exists),
                )
            )
    remaining, last_end = [], 0
    for m in matches:
        remaining.append(text[last_end:m.start()])
        last_end = m.end()
    remaining.append(text[last_end:])
    return goals, "".join(remaining)


def parse_instruction(text: str, known_objects: Iterable[str]) -> GoalGraph:
    """Parse a natural-language instruction into a GoalGraph.

    `known_objects` is the closed set of object ids valid for the scene
    this instruction will run in (e.g. {"red_mug", "blue_bowl",
    "medicine_bottle", "glass"}) -- required to disambiguate phrases like
    "bowl" vs "blue bowl" when both could otherwise match, and to resolve
    conditional clauses' trigger objects.
    """
    known_objects = list(known_objects)
    conditional_goals, remaining_text = _extract_conditional_goals(text, known_objects)

    clauses = [c.strip(" ,.") for c in _CLAUSE_SEP.split(remaining_text.strip()) if c.strip(" ,.")]
    if not clauses and not conditional_goals:
        raise ValueError(f"could not split any clauses from instruction: {text!r}")

    goals: list[Goal] = list(conditional_goals)
    constraints: list[Constraint] = []
    order_counter = 0
    for clause in clauses:
        kind, phrases, order_marker = _classify_clause(clause)
        for phrase in phrases:
            obj_id = _resolve_object(phrase, known_objects)
            if kind == "goal":
                priority = 0
                if order_marker is not None:
                    priority = order_counter
                    order_counter += 1
                goals.append(
                    Goal(id=f"place_{obj_id}", predicate="on_tray", target_object=obj_id, priority=priority)
                )
            elif kind == "orient":
                constraints.append(
                    Constraint(id=f"keep_{obj_id}_upright", kind="maintain_orientation", target_object=obj_id)
                )
            else:
                constraints.append(
                    Constraint(id=f"dont_move_{obj_id}", kind="never_move", target_object=obj_id)
                )

    return GoalGraph(instruction_text=text, goals=tuple(goals), constraints=tuple(constraints))
