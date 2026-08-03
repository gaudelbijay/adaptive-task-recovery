"""Tests for atr.evaluation.splits (D-044) -- no mani_skill needed at all
(unlike test_instruction_parser.py, which needs it transitively to import
replicacad_example()/replicacad_humanoid_example()). This file only
touches atr.language.instruction_parser/goal_graph, both mani_skill-free,
so it runs in the fast/no-simulator CI tier (D-043) for real, not just in
theory.
"""

import pytest

from atr.evaluation.splits import (
    CANONICAL_OBJECTS,
    HELD_OUT_COMPOSITION,
    HELD_OUT_PARAPHRASE,
    SPLITS,
    TRAIN,
    all_specs,
)
from atr.language.goal_graph import canonical_example
from atr.language.instruction_parser import parse_instruction


def _semantics(graph):
    goals = frozenset((g.predicate, g.target_object) for g in graph.goals)
    constraints = frozenset((c.kind, c.target_object) for c in graph.constraints)
    return goals, constraints


class TestRegistryStructure:
    def test_all_specs_covers_every_split(self):
        assert all_specs() == TRAIN + HELD_OUT_PARAPHRASE + HELD_OUT_COMPOSITION

    def test_splits_dict_matches_the_tuples(self):
        assert SPLITS["train"] == TRAIN
        assert SPLITS["held_out_paraphrase"] == HELD_OUT_PARAPHRASE
        assert SPLITS["held_out_composition"] == HELD_OUT_COMPOSITION

    def test_every_spec_tagged_with_its_own_split_name(self):
        for split_name, specs in SPLITS.items():
            assert all(spec.split == split_name for spec in specs)


class TestEverySpecParsesWithoutRaising:
    """The minimum bar: whatever else is true, every spec in every split
    must be parseable against its own known_objects -- a spec that
    doesn't even parse would be a broken split, not a held-out example."""

    @pytest.mark.parametrize("spec", all_specs(), ids=lambda s: s.instruction_text[:40])
    def test_parses(self, spec):
        graph = parse_instruction(spec.instruction_text, spec.known_objects)
        assert graph.goals or graph.constraints


class TestCanonicalSplitsMatchTheHandAuthoredGraph:
    """Every canonical-object spec (train + the 3 canonical held-out
    paraphrases) should parse to the same semantics as canonical_example()
    -- checked here with zero mani_skill dependency, since canonical_example()
    lives in atr.language.goal_graph. The 2 ReplicaCAD-object paraphrases and
    the held-out composition are still checked against their own
    hand-authored graphs in test_instruction_parser.py, which does need
    mani_skill (replicacad_example() lives in spike env files)."""

    def test_every_canonical_object_spec_matches_canonical_example(self):
        expected = _semantics(canonical_example())
        canonical_specs = [s for s in all_specs() if s.known_objects == CANONICAL_OBJECTS]
        assert len(canonical_specs) == 4  # 1 train + 3 held-out paraphrases, not silently 0
        for spec in canonical_specs:
            parsed = parse_instruction(spec.instruction_text, spec.known_objects)
            assert _semantics(parsed) == expected, spec.instruction_text
