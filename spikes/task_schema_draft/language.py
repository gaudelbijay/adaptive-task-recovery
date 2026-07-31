"""Stage 2 of docs/00-project-overview.md's build-up order: parse an actual
instruction sentence into a GoalGraph, instead of writing one by hand.

Deliberately a controlled grammar, not open-ended NLU, per
docs/04-benchmark-environment.md: "Language templates should support
conjunction, ordering, exclusion, conditional goals, and preferences. Hold
out paraphrases and compositions..." This covers conjunction and exclusion
(the two forms already used by every existing hand-authored GoalGraph in
this directory); ordering/priority and conditional goals are not
implemented yet -- no existing instruction uses them, and adding grammar for
them without a test case to drive it would be speculative.

Object names are resolved against a caller-supplied vocabulary (the set of
object ids that actually exist in a given scene), not guessed from open
vocabulary -- this project already assumes a closed, object-centric world
(docs/00 "Scope (v1)"), so the parser does too. An unrecognized clause
raises rather than being silently dropped: silently ignoring a "do not
move X" clause would be exactly the kind of intent violation this project
exists to catch.
"""

from __future__ import annotations

import re
from typing import Iterable

from task_schema_draft.goal_graph import Constraint, Goal, GoalGraph

_CLAUSE_VERBS = r"put|place|keep|do not|don't|dont|never|leave"

_CLAUSE_SEP = re.compile(
    rf"(?:,\s*(?:and\s+)?|\.\s*|;\s*|\s+and\s+)(?=(?:{_CLAUSE_VERBS})\b)",
    re.IGNORECASE,
)

_GOAL_RE = re.compile(r"^(?:put|place)\s+(.+?)\s+on\s+the\s+\w+$", re.IGNORECASE)
_ORIENT_RE = re.compile(r"^keep\s+(?:the\s+)?(.+?)\s+upright$", re.IGNORECASE)
_NEVER_MOVE_RE = re.compile(
    r"^(?:do not|don't|dont|never)\s+(?:move|touch)\s+(?:the\s+)?(.+?)$",
    re.IGNORECASE,
)
_LEAVE_ALONE_RE = re.compile(r"^leave\s+(?:the\s+)?(.+?)\s+alone$", re.IGNORECASE)


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


def _classify_clause(clause: str) -> tuple[str, list[str]]:
    if m := _GOAL_RE.match(clause):
        return "goal", _split_object_list(m.group(1))
    if m := _ORIENT_RE.match(clause):
        return "orient", [_clean_phrase(m.group(1))]
    if m := _NEVER_MOVE_RE.match(clause):
        return "never_move", [_clean_phrase(m.group(1))]
    if m := _LEAVE_ALONE_RE.match(clause):
        return "never_move", [_clean_phrase(m.group(1))]
    raise ValueError(f"unrecognized clause: {clause!r}")


def parse_instruction(text: str, known_objects: Iterable[str]) -> GoalGraph:
    """Parse a natural-language instruction into a GoalGraph.

    `known_objects` is the closed set of object ids valid for the scene
    this instruction will run in (e.g. {"red_mug", "blue_bowl",
    "medicine_bottle", "glass"}) -- required to disambiguate phrases like
    "bowl" vs "blue bowl" when both could otherwise match.
    """
    known_objects = list(known_objects)
    clauses = [c.strip(" ,.") for c in _CLAUSE_SEP.split(text.strip()) if c.strip(" ,.")]
    if not clauses:
        raise ValueError(f"could not split any clauses from instruction: {text!r}")

    goals: list[Goal] = []
    constraints: list[Constraint] = []
    for clause in clauses:
        kind, phrases = _classify_clause(clause)
        for phrase in phrases:
            obj_id = _resolve_object(phrase, known_objects)
            if kind == "goal":
                goals.append(Goal(id=f"place_{obj_id}", predicate="on_tray", target_object=obj_id))
            elif kind == "orient":
                constraints.append(
                    Constraint(id=f"keep_{obj_id}_upright", kind="maintain_orientation", target_object=obj_id)
                )
            else:
                constraints.append(
                    Constraint(id=f"dont_move_{obj_id}", kind="never_move", target_object=obj_id)
                )

    return GoalGraph(instruction_text=text, goals=tuple(goals), constraints=tuple(constraints))
