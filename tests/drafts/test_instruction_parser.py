"""Tests for instruction_parser.py -- stage 2 of docs/00-project-overview.md's
build-up order ("parse an actual instruction sentence into a GoalGraph,
instead of writing one by hand").

Checks three things: (1) the parser reproduces every existing hand-authored
GoalGraph from its own instruction_text, (2) it generalizes to paraphrases
and compositions it has never seen (the explicit requirement in
docs/04-benchmark-environment.md), and (3) it fails loudly rather than
silently dropping a clause it doesn't understand -- silently dropping a
"do not move X" constraint would defeat the entire point of this project.
"""

import pytest

pytest.importorskip("mani_skill")  # replicacad_example/_humanoid pull in mani_skill env classes

from atr.language.goal_graph import canonical_example  # noqa: E402
from task_schema_draft.instruction_parser import parse_instruction  # noqa: E402
from task_schema_draft.tidy_up_env_replicacad import replicacad_example  # noqa: E402
from task_schema_draft.tidy_up_env_replicacad_humanoid import (  # noqa: E402
    replicacad_humanoid_example,
)

CANONICAL_OBJECTS = {"red_mug", "blue_bowl", "medicine_bottle", "glass"}
REPLICACAD_OBJECTS = {"potted_meat_can", "bowl", "cracker_box", "master_chef_can"}
HUMANOID_OBJECTS = {"potted_meat_can", "master_chef_can", "cracker_box", "bowl"}


def _semantics(graph):
    goals = frozenset((g.predicate, g.target_object) for g in graph.goals)
    constraints = frozenset((c.kind, c.target_object) for c in graph.constraints)
    return goals, constraints


class TestReproducesHandAuthoredGraphs:
    def test_canonical_example(self):
        expected = canonical_example()
        parsed = parse_instruction(expected.instruction_text, CANONICAL_OBJECTS)
        assert _semantics(parsed) == _semantics(expected)

    def test_replicacad_example(self):
        expected = replicacad_example()
        parsed = parse_instruction(expected.instruction_text, REPLICACAD_OBJECTS)
        assert _semantics(parsed) == _semantics(expected)

    def test_replicacad_humanoid_example(self):
        expected = replicacad_humanoid_example()
        parsed = parse_instruction(expected.instruction_text, HUMANOID_OBJECTS)
        assert _semantics(parsed) == _semantics(expected)


class TestHeldOutParaphrases:
    """None of these exact strings were used to write instruction_parser.py's regexes
    against -- they're phrased differently from every instruction_text
    above (different verb, negation form, conjunction style, punctuation)."""

    def test_different_verb_and_negation(self):
        text = "Place the red mug and the blue bowl on the tray. Keep the medicine upright. Never move the glass."
        parsed = parse_instruction(text, CANONICAL_OBJECTS)
        assert _semantics(parsed) == _semantics(canonical_example())

    def test_no_comma_all_and(self):
        text = "Put the red mug and blue bowl on the tray and keep the medicine upright and don't move the glass"
        parsed = parse_instruction(text, CANONICAL_OBJECTS)
        assert _semantics(parsed) == _semantics(canonical_example())

    def test_oxford_comma_three_goal_objects(self):
        text = (
            "Put the potted meat can, the bowl, and the cracker box on the tray, "
            "keep the cracker box upright, and do not move the master chef can."
        )
        parsed = parse_instruction(text, REPLICACAD_OBJECTS)
        assert ("on_tray", "potted_meat_can") in _semantics(parsed)[0]
        assert ("on_tray", "bowl") in _semantics(parsed)[0]
        assert ("on_tray", "cracker_box") in _semantics(parsed)[0]

    def test_leave_alone_phrasing(self):
        text = "Put the potted meat can and the bowl on the table, keep the cracker box upright, and leave the master chef can alone."
        parsed = parse_instruction(text, REPLICACAD_OBJECTS)
        assert _semantics(parsed) == _semantics(replicacad_example())

    def test_clause_order_reversed(self):
        text = "Do not move the glass. Keep the medicine upright. Put the red mug and blue bowl on the tray."
        parsed = parse_instruction(text, CANONICAL_OBJECTS)
        assert _semantics(parsed) == _semantics(canonical_example())


class TestHeldOutComposition:
    """A new instruction, recombining objects across scenes, that was never
    written as a hand-authored GoalGraph anywhere in this project."""

    def test_novel_object_combination(self):
        text = (
            "Put the master chef can and the glass on the tray, keep the medicine "
            "upright, and do not move the cracker box."
        )
        known = {"master_chef_can", "glass", "medicine_bottle", "cracker_box"}
        parsed = parse_instruction(text, known)
        goals, constraints = _semantics(parsed)
        assert goals == frozenset({("on_tray", "master_chef_can"), ("on_tray", "glass")})
        assert constraints == frozenset(
            {("maintain_orientation", "medicine_bottle"), ("never_move", "cracker_box")}
        )


class TestOrderingAndPriority:
    """D-026: order markers (first/then/next/finally/...) assign
    Goal.priority in order of appearance; goals without a marker default to
    priority=0, unchanged from before this was added."""

    def test_first_then_assigns_sequential_priority(self):
        text = (
            "First put the red mug on the tray, then put the blue bowl on "
            "the tray, keep the medicine upright, and do not move the glass."
        )
        parsed = parse_instruction(text, CANONICAL_OBJECTS)
        priorities = {g.target_object: g.priority for g in parsed.goals}
        assert priorities["red_mug"] < priorities["blue_bowl"]

    def test_no_marker_keeps_default_priority_zero(self):
        parsed = parse_instruction(canonical_example().instruction_text, CANONICAL_OBJECTS)
        assert all(g.priority == 0 for g in parsed.goals)

    def test_next_marker_also_recognized(self):
        # Order markers apply to goal clauses only -- Constraint has no
        # priority field, so an order marker before "keep X upright" would
        # have nowhere to go; this only exercises "next" on goals.
        text = "First put the red mug on the tray, next put the blue bowl on the tray, keep the medicine upright, and do not move the glass."
        parsed = parse_instruction(text, CANONICAL_OBJECTS)
        priorities = {g.target_object: g.priority for g in parsed.goals}
        assert priorities["red_mug"] < priorities["blue_bowl"]


class TestConditionalGoals:
    """D-026 (PROPOSED, not yet reviewed): "if X is Y, do Z instead" sets
    Goal.condition = (trigger_object_id, required_exists)."""

    def test_negative_trigger_state(self):
        text = "Put the red mug on the tray. If the blue bowl is destroyed, put the backup bowl on the tray instead."
        known = {"red_mug", "blue_bowl", "backup_bowl"}
        parsed = parse_instruction(text, known)
        by_object = {g.target_object: g for g in parsed.goals}
        assert by_object["backup_bowl"].condition == ("blue_bowl", False)
        assert by_object["red_mug"].condition is None

    def test_positive_trigger_state(self):
        text = "If the cracker box is present, put the red mug on the tray."
        known = {"red_mug", "cracker_box"}
        parsed = parse_instruction(text, known)
        assert parsed.goals[0].condition == ("cracker_box", True)

    def test_comma_then_phrasing(self):
        text = "If the blue bowl is destroyed, then put the backup bowl on the tray instead."
        known = {"blue_bowl", "backup_bowl"}
        parsed = parse_instruction(text, known)
        assert parsed.goals[0].condition == ("blue_bowl", False)

    def test_conditional_clause_embedded_mid_instruction(self):
        """The real bug this grammar had to work around: a comma right
        before "put" ordinarily starts a new clause (per every other test
        in this file), but here it's internal to one conditional clause and
        must not be split apart from it."""
        text = (
            "Put the red mug on the tray, if the blue bowl is destroyed put "
            "the backup bowl on the tray instead, keep the medicine upright, "
            "and do not move the glass."
        )
        known = {"red_mug", "blue_bowl", "backup_bowl", "medicine_bottle", "glass"}
        parsed = parse_instruction(text, known)
        by_object = {g.target_object: g for g in parsed.goals}
        assert by_object["backup_bowl"].condition == ("blue_bowl", False)
        assert by_object["red_mug"].condition is None
        assert {(c.kind, c.target_object) for c in parsed.constraints} == {
            ("maintain_orientation", "medicine_bottle"), ("never_move", "glass"),
        }


class TestFailsLoudlyRatherThanSilently:
    def test_unrecognized_clause_raises(self):
        with pytest.raises(ValueError, match="unrecognized clause"):
            parse_instruction(
                "Put the red mug on the tray, please water the glass.", CANONICAL_OBJECTS
            )

    def test_ambiguous_object_raises(self):
        # "red" has no exact match but is a subset of both candidates -- a
        # real ambiguity, unlike "bowl" vs "blue_bowl" (an exact match to
        # "bowl" exists, so that case resolves unambiguously).
        with pytest.raises(ValueError, match="ambiguous"):
            parse_instruction(
                "Put the red on the tray.", {"red_mug", "red_cup"}
            )

    def test_unknown_object_raises(self):
        with pytest.raises(ValueError, match="unrecognized object phrase"):
            parse_instruction(
                "Put the red mug on the tray.", {"blue_bowl"}
            )
