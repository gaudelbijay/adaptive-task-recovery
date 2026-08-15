"""H4 (compositional generalization): factorized goal and change
representations transfer better to unseen goal-change combinations than a
monolithic policy.

D-019/D-038 already established that `instruction_parser.py`'s
controlled-grammar parser -- a genuinely factorized representation, since
it compiles each clause into its own `Goal`/`Constraint` independent of
the others -- generalizes correctly to held-out paraphrases and a
held-out object composition (`atr.evaluation.splits.SPLITS`, D-044).
What was never built or run: the actual *comparative* half of H4's claim,
against an explicit non-factorized alternative. `train_monolithic_lookup()`
is that alternative -- an instruction system with no compositional
mechanism at all, only exact-string memorization of what it was shown
during training. A fair baseline in the sense that it gets full credit
for the training instructions it actually saw (built via the real parser,
so its training-set answers are exactly as correct as the factorized
parser's own); its known limitation isn't imagined, it's structural: a
paraphrase or novel object composition is, by construction, a string the
lookup has never seen, so it can only fail to answer, never generalize.

`compare_factorized_vs_monolithic()` runs both across a fixed,
independently-verified evaluation set (ground truth semantics copied
verbatim from already-validated assertions in `test_instruction_parser.py`
and `test_splits.py` -- not the parser's own output, since scoring the
parser against itself would be circular) spanning all three splits
(`train`, `held_out_paraphrase`, `held_out_composition`).

D-080 adds a materially stronger monolithic baseline: character-trigram
nearest-neighbor retrieval. It learns an indivisible text-to-whole-graph
mapping and always returns the graph attached to the closest training text.
This removes the exact lookup's main confound: the retriever transfers the
canonical graph across every held-out paraphrase, but still cannot construct a
new composition because it can only retrieve whole graphs it has seen.

D-081 expands the one-train/one-composition comparison into a matrix of four
training and four semantically disjoint held-out role recombinations. Objects
are familiar on both sides; only their assignments to goal, orientation, and
protection roles change. Ground truth is built from those declared roles rather
than parser output.

D-117 broadens D-081's 4-train/4-held-out hand-picked matrix into
`full_role_matrix_cases()` -- every possible goal-pair over a 6-object pool
(180 cases: 96 train, 84 held-out), split with a checked guarantee that no
held-out goal-pair ever appeared as a goal-pair in training. Result:
identical to D-081's qualitative finding (factorized 100%/100%, both
monolithic baselines 100%/0%) but now backed by the full combinatorial
space a rule-based parser with no sampling variance can actually offer,
not a hand-picked sample of it -- the real value of the larger sweep is
stress-testing `_resolve_object()`'s word-set object-matching logic against
many more distinct strings, not statistical confidence.

Attempting an intervention-mechanism axis for H4 first found a real
scoping problem instead of guessing past it: every intervention kind in
every env variant in this project (`bowl_destroyed`, `temporary_obstacle`,
`resource_contention`, `resource_contention_temporary`, `chef_can_destroyed`)
threatens exactly one specific goal each; no env has two goals each
independently threatened by a different intervention kind, so there is no
real goal-by-intervention cross product to hold a combination out from at
the simulator level. The language axis (`atr.evaluation.splits`'s
`InstructionSpec`s, D-044) has that real compositional structure already
-- known objects recombined into instructions never literally seen
together -- so this module tests H4 there instead, a smaller and more
honest scope than assuming the intervention axis would work.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, permutations
from math import sqrt

from atr.language.goal_graph import GoalGraph, canonical_example
from atr.language.instruction_parser import parse_instruction

Semantics = tuple[frozenset, frozenset]


def semantics(graph: GoalGraph) -> Semantics:
    """(goals, constraints) as comparable frozensets of (kind, target_object)
    pairs -- ignores incidental details (priority, condition, dependency
    wiring) the same way every existing parser-correctness test in this
    project already does (`test_instruction_parser.py`'s `_semantics()`),
    reused here rather than redefined so "correct" means the same thing it
    always has."""
    goals = frozenset((g.predicate, g.target_object) for g in graph.goals)
    constraints = frozenset((c.kind, c.target_object) for c in graph.constraints)
    return goals, constraints


# Ground truth for the held-out composition spec, copied verbatim from
# test_instruction_parser.py's TestHeldOutComposition (already-validated,
# zero mani_skill dependency -- the same "reuse, don't re-derive" pattern
# atr.evaluation.splits itself documents using for its instruction strings).
_HELD_OUT_COMPOSITION_GROUND_TRUTH: Semantics = (
    frozenset({("on_tray", "master_chef_can"), ("on_tray", "glass")}),
    frozenset({("maintain_orientation", "medicine_bottle"), ("never_move", "cracker_box")}),
)


@dataclass(frozen=True)
class EvalCase:
    instruction_text: str
    known_objects: frozenset[str]
    split: str
    ground_truth: Semantics


def canonical_eval_cases() -> tuple[EvalCase, ...]:
    """Every spec in `atr.evaluation.splits` whose ground truth is available
    without a mani_skill dependency: the four `CANONICAL_OBJECTS` specs
    (train + 3 held-out paraphrases, checked against `canonical_example()`,
    same scope `test_splits.py::TestCanonicalSplitsMatchTheHandAuthoredGraph`
    already uses) plus the one held-out-composition spec (checked against
    the ground truth above). Deliberately excludes the ReplicaCAD/humanoid
    train and paraphrase specs -- their hand-authored ground truth
    (`replicacad_example()` etc.) lives in mani_skill-dependent spike files,
    and this comparison doesn't need the simulator at all to make its point."""
    from atr.evaluation.splits import HELD_OUT_COMPOSITION, HELD_OUT_PARAPHRASE, TRAIN, CANONICAL_OBJECTS

    canonical_truth = semantics(canonical_example())
    cases = [
        EvalCase(spec.instruction_text, spec.known_objects, spec.split, canonical_truth)
        for spec in TRAIN + HELD_OUT_PARAPHRASE
        if spec.known_objects == CANONICAL_OBJECTS
    ]
    cases += [
        EvalCase(spec.instruction_text, spec.known_objects, spec.split, _HELD_OUT_COMPOSITION_GROUND_TRUTH)
        for spec in HELD_OUT_COMPOSITION
    ]
    return tuple(cases)


def _matrix_case(
    goal_objects: tuple[str, str], orient_object: str, protected_object: str, split: str,
) -> EvalCase:
    """Build one independently labelled role assignment for the H4 matrix."""
    known_objects = frozenset((*goal_objects, orient_object, protected_object))
    readable = {obj: obj.replace("_", " ") for obj in known_objects}
    text = (
        f"Put the {readable[goal_objects[0]]} and the {readable[goal_objects[1]]} on the tray, "
        f"keep the {readable[orient_object]} upright, and do not move the "
        f"{readable[protected_object]}."
    )
    truth = (
        frozenset(("on_tray", obj) for obj in goal_objects),
        frozenset({
            ("maintain_orientation", orient_object),
            ("never_move", protected_object),
        }),
    )
    return EvalCase(text, known_objects, split, truth)


def compositional_matrix_cases() -> tuple[EvalCase, ...]:
    """A larger role-recombination split with disjoint semantic graphs.

    D-079/D-080 had one training graph and one held-out composition. This
    matrix assigns familiar objects to goal/orientation/protection roles in
    four training and four held-out combinations. Ground truth is constructed
    directly from those declared roles, never from parser output.
    """
    train = (
        _matrix_case(("red_mug", "blue_bowl"), "medicine_bottle", "glass", "train"),
        _matrix_case(("cracker_box", "master_chef_can"), "glass", "medicine_bottle", "train"),
        _matrix_case(("red_mug", "cracker_box"), "master_chef_can", "blue_bowl", "train"),
        _matrix_case(("blue_bowl", "medicine_bottle"), "cracker_box", "master_chef_can", "train"),
    )
    held_out = (
        _matrix_case(("master_chef_can", "glass"), "medicine_bottle", "cracker_box", "held_out_composition"),
        _matrix_case(("medicine_bottle", "glass"), "red_mug", "blue_bowl", "held_out_composition"),
        _matrix_case(("blue_bowl", "master_chef_can"), "glass", "red_mug", "held_out_composition"),
        _matrix_case(("cracker_box", "medicine_bottle"), "blue_bowl", "glass", "held_out_composition"),
    )
    return train + held_out


_MATRIX_OBJECTS = ("red_mug", "blue_bowl", "medicine_bottle", "glass", "cracker_box", "master_chef_can")


def full_role_matrix_cases(objects: tuple[str, ...] = _MATRIX_OBJECTS) -> tuple[EvalCase, ...]:
    """D-117: exhaustive version of `compositional_matrix_cases()`'s 4
    train / 4 held-out hand-picked cases. D-081's matrix was systematic in
    spirit but small enough that "generalizes" rested on 4 examples; this
    sweeps every possible goal-pair over `objects` instead, so the claim is
    backed by the full combinatorial space a 6-object pool actually allows,
    not a hand-picked sample of it.

    Split by goal-pair, not by full 4-object case: `combinations(objects, 2)`
    enumerates every possible *pair of goal targets* once, alternately
    assigned to train/held-out (`[0::2]`/`[1::2]`) so both splits cover the
    object pool evenly rather than an arbitrary prefix cut skewing which
    objects land where. This guarantees -- checked, not assumed -- that no
    goal-pair used in a held-out case was ever a goal-pair in a train case,
    which is this project's own stated definition of a held-out composition
    (`atr.evaluation.splits`'s "known objects recombined into instructions
    never literally seen together"). For each goal-pair, every
    (orient, protect) assignment of the remaining objects is included, not
    just one -- the actual stress-test value of a larger matrix is exercising
    the parser's object-resolution logic (`_resolve_object`'s word-set
    matching) against many more distinct instruction strings, since a
    deterministic rule-based parser has no sampling variance to average
    over; the risk a bigger matrix can actually catch is an unanticipated
    string-matching edge case, not statistical noise.
    """
    all_pairs = tuple(combinations(sorted(objects), 2))
    train_pairs = all_pairs[0::2]
    held_out_pairs = all_pairs[1::2]

    def cases_for(pairs: tuple[tuple[str, str], ...], split: str) -> list[EvalCase]:
        cases = []
        for goal_pair in pairs:
            remaining = [obj for obj in objects if obj not in goal_pair]
            for orient_object, protected_object in permutations(remaining, 2):
                cases.append(_matrix_case(goal_pair, orient_object, protected_object, split))
        return cases

    return tuple(cases_for(train_pairs, "train") + cases_for(held_out_pairs, "held_out_composition"))


def train_monolithic_lookup(train_cases: list[EvalCase]) -> dict[str, Semantics]:
    """The non-factorized alternative: exact-instruction-string memorization,
    with no mechanism for handling any string it wasn't shown. Its
    training-set answers come from the real parser (a fair baseline gets
    full credit on what it actually saw), but nothing about *how* those
    answers were produced is retained or reused for a new string."""
    return {case.instruction_text: semantics(parse_instruction(case.instruction_text, case.known_objects))
            for case in train_cases}


def monolithic_predict(lookup: dict[str, Semantics], instruction_text: str) -> Semantics | None:
    """`None` on any instruction text not seen verbatim during training --
    by construction, not a bug to fix: a monolithic memorizer has no other
    behavior available for an unseen string."""
    return lookup.get(instruction_text)


@dataclass(frozen=True)
class RetrievalExample:
    features: Counter[str]
    target: Semantics


def _character_ngrams(text: str, n: int = 3) -> Counter[str]:
    """Normalized character n-grams: a surface representation only.

    It has no object/clause/goal slots and therefore no compositional
    mechanism. Character rather than word n-grams let paraphrases share useful
    lexical fragments without importing the factorized parser's vocabulary.
    """
    normalized = " ".join(text.lower().split())
    padded = f"  {normalized}  "
    return Counter(padded[i:i + n] for i in range(len(padded) - n + 1))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def train_monolithic_retriever(train_cases: list[EvalCase]) -> tuple[RetrievalExample, ...]:
    """Fit a stronger non-factorized baseline from surface text to semantics.

    Unlike exact lookup, it always transfers the target of the most similar
    training sentence. It can therefore handle surface paraphrases when their
    semantics match a training example, but it cannot assemble parts of
    different targets into a new graph.
    """
    return tuple(
        RetrievalExample(
            _character_ngrams(case.instruction_text),
            semantics(parse_instruction(case.instruction_text, case.known_objects)),
        )
        for case in train_cases
    )


def monolithic_retrieval_predict(
    model: tuple[RetrievalExample, ...], instruction_text: str,
) -> Semantics | None:
    if not model:
        return None
    query = _character_ngrams(instruction_text)
    return max(model, key=lambda example: _cosine(query, example.features)).target


@dataclass(frozen=True)
class CompositionalGeneralizationResult:
    """Per-split accuracy for both representations, plus the raw per-case
    correctness so a caller can see exactly which cases each one got right,
    not just an aggregate."""

    factorized_correct_by_split: dict[str, int]
    monolithic_correct_by_split: dict[str, int]
    retrieval_correct_by_split: dict[str, int]
    total_by_split: dict[str, int]
    factorized_case_results: tuple[bool, ...]
    monolithic_case_results: tuple[bool, ...]
    retrieval_case_results: tuple[bool, ...]


def compare_factorized_vs_monolithic(
    eval_cases: tuple[EvalCase, ...] | None = None,
) -> CompositionalGeneralizationResult:
    """Runs H4's actual comparative claim: the real, factorized
    `instruction_parser.py` versus a monolithic exact-string memorizer,
    across every split in `canonical_eval_cases()` (train, held-out
    paraphrase, held-out composition). The monolithic baseline is trained
    only on the `train`-split cases within `eval_cases` -- it never sees a
    held-out instruction's text during "training," the same disjointness
    every other split-based comparison in this project (D-069, D-076)
    requires."""
    cases = eval_cases if eval_cases is not None else canonical_eval_cases()
    train_cases = [case for case in cases if case.split == "train"]
    lookup = train_monolithic_lookup(train_cases)
    retriever = train_monolithic_retriever(train_cases)

    factorized_correct_by_split: dict[str, int] = {}
    monolithic_correct_by_split: dict[str, int] = {}
    retrieval_correct_by_split: dict[str, int] = {}
    total_by_split: dict[str, int] = {}
    factorized_case_results = []
    monolithic_case_results = []
    retrieval_case_results = []

    for case in cases:
        total_by_split[case.split] = total_by_split.get(case.split, 0) + 1

        parsed = semantics(parse_instruction(case.instruction_text, case.known_objects))
        factorized_ok = parsed == case.ground_truth
        factorized_case_results.append(factorized_ok)
        factorized_correct_by_split[case.split] = factorized_correct_by_split.get(case.split, 0) + factorized_ok

        predicted = monolithic_predict(lookup, case.instruction_text)
        monolithic_ok = predicted == case.ground_truth
        monolithic_case_results.append(monolithic_ok)
        monolithic_correct_by_split[case.split] = monolithic_correct_by_split.get(case.split, 0) + monolithic_ok

        retrieved = monolithic_retrieval_predict(retriever, case.instruction_text)
        retrieval_ok = retrieved == case.ground_truth
        retrieval_case_results.append(retrieval_ok)
        retrieval_correct_by_split[case.split] = (
            retrieval_correct_by_split.get(case.split, 0) + retrieval_ok
        )

    return CompositionalGeneralizationResult(
        factorized_correct_by_split=factorized_correct_by_split,
        monolithic_correct_by_split=monolithic_correct_by_split,
        retrieval_correct_by_split=retrieval_correct_by_split,
        total_by_split=total_by_split,
        factorized_case_results=tuple(factorized_case_results),
        monolithic_case_results=tuple(monolithic_case_results),
        retrieval_case_results=tuple(retrieval_case_results),
    )
