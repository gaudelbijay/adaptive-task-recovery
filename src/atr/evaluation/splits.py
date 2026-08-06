"""Instruction-level dataset splits (D-044), plus intervention-level splits
(D-059). docs/04-benchmark-environment.md requires holding out
"paraphrases and compositions"; docs/10-evaluation-and-benchmarks.md's
statistical protocol requires "predeclare primary metrics and splits."
Both existed only as literal strings inside
`tests/drafts/test_instruction_parser.py`'s test-function bodies until
this -- real, validated evidence, but with no way for anything else
(the evaluation harness, D-042; a future benchmark runner) to ask "give
me every held-out-paraphrase spec" programmatically.

`InterventionSpec`/`INTERVENTION_SPLITS` (D-059) are the same idea for a
different axis: docs/10's "seen versus held-out goal-change combinations"
ablation needs more than 2 total intervention kinds to hold one out at
all meaningfully. `tidy_up_env.py`'s `resource_contention`/
`resource_contention_temporary` (D-059) are the first kinds
mechanistically different enough from `bowl_destroyed`/
`temporary_obstacle` (contingent on episode progress, not a blind timer)
to be a genuine held-out test, not just a differently-named copy of the
same mechanism.

Pure data, no simulator/env dependency by design: an `InstructionSpec` is
just `(instruction_text, known_objects, split)`. Deliberately does NOT
carry each spec's *expected* `GoalGraph` -- that would mean importing the
spike env files that define `replicacad_example()`/
`replicacad_humanoid_example()` (mani_skill-dependent), which would point
promoted code back at spike code, the same dependency-direction problem
every promotion since D-037 has avoided. Checking parser output against
those hand-authored graphs stays exactly where it's already exercised
(`tests/drafts/test_instruction_parser.py`); this module is for
enumerating *what to run*, not for asserting correctness itself.

Every string below is copied verbatim from already-validated test cases
(`test_instruction_parser.py`'s `TestHeldOutParaphrases`/
`TestHeldOutComposition`, and each env's own hand-authored
`instruction_text`) -- nothing here is new/unvalidated content.
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_OBJECTS = frozenset({"red_mug", "blue_bowl", "medicine_bottle", "glass"})
REPLICACAD_OBJECTS = frozenset({"potted_meat_can", "bowl", "cracker_box", "master_chef_can"})
HUMANOID_OBJECTS = frozenset({"potted_meat_can", "master_chef_can", "cracker_box", "bowl"})


@dataclass(frozen=True)
class InstructionSpec:
    """One instruction to parse, and the closed object vocabulary to
    resolve it against -- exactly `parse_instruction()`'s own two
    arguments, plus which split it belongs to."""

    instruction_text: str
    known_objects: frozenset[str]
    split: str  # "train" | "held_out_paraphrase" | "held_out_composition"


TRAIN: tuple[InstructionSpec, ...] = (
    InstructionSpec(
        "Put the red mug and blue bowl on the tray, keep the medicine "
        "upright, and do not move the glass.",
        CANONICAL_OBJECTS, "train",
    ),
    InstructionSpec(
        "Put the potted meat can and the bowl on the table, keep the "
        "cracker box upright, and do not move the master chef can.",
        REPLICACAD_OBJECTS, "train",
    ),
    InstructionSpec(
        "Put the potted meat can and the master chef can on the tray, "
        "keep the cracker box upright, and do not move the bowl.",
        HUMANOID_OBJECTS, "train",
    ),
)

HELD_OUT_PARAPHRASE: tuple[InstructionSpec, ...] = (
    InstructionSpec(
        "Place the red mug and the blue bowl on the tray. Keep the medicine "
        "upright. Never move the glass.",
        CANONICAL_OBJECTS, "held_out_paraphrase",
    ),
    InstructionSpec(
        "Put the red mug and blue bowl on the tray and keep the medicine "
        "upright and don't move the glass",
        CANONICAL_OBJECTS, "held_out_paraphrase",
    ),
    InstructionSpec(
        "Do not move the glass. Keep the medicine upright. Put the red mug "
        "and blue bowl on the tray.",
        CANONICAL_OBJECTS, "held_out_paraphrase",
    ),
    InstructionSpec(
        "Put the potted meat can, the bowl, and the cracker box on the "
        "tray, keep the cracker box upright, and do not move the master "
        "chef can.",
        REPLICACAD_OBJECTS, "held_out_paraphrase",
    ),
    InstructionSpec(
        "Put the potted meat can and the bowl on the table, keep the "
        "cracker box upright, and leave the master chef can alone.",
        REPLICACAD_OBJECTS, "held_out_paraphrase",
    ),
)

HELD_OUT_COMPOSITION: tuple[InstructionSpec, ...] = (
    InstructionSpec(
        "Put the master chef can and the glass on the tray, keep the "
        "medicine upright, and do not move the cracker box.",
        frozenset({"master_chef_can", "glass", "medicine_bottle", "cracker_box"}),
        "held_out_composition",
    ),
)

SPLITS: dict[str, tuple[InstructionSpec, ...]] = {
    "train": TRAIN,
    "held_out_paraphrase": HELD_OUT_PARAPHRASE,
    "held_out_composition": HELD_OUT_COMPOSITION,
}


def all_specs() -> tuple[InstructionSpec, ...]:
    """Every spec across every split, in split order (train first)."""
    return TRAIN + HELD_OUT_PARAPHRASE + HELD_OUT_COMPOSITION


@dataclass(frozen=True)
class InterventionSpec:
    """One (env, intervention_kind) combination to evaluate, and which
    split it belongs to -- the intervention-axis counterpart to
    InstructionSpec above. Added D-059, once a 3rd intervention_kind
    (`resource_contention`/its matched `resource_contention_temporary`
    counterpart, `tidy_up_env.py`) existed to hold out at all: with only
    2 kinds total, "train on some, hold out one, unseen" wasn't
    constructible before this."""

    env_id: str
    intervention_kind: str
    split: str  # "train" | "held_out_intervention"


INTERVENTION_TRAIN: tuple[InterventionSpec, ...] = (
    InterventionSpec("TidyUp-v1", "bowl_destroyed", "train"),
    InterventionSpec("TidyUp-v1", "temporary_obstacle", "train"),
)

# D-059's contingent-on-progress mechanism, genuinely different from
# bowl_destroyed's blind timer -- held out, not trained on, so a policy
# tuned only against the two timer-based interventions above can be
# checked against a kind it's never seen.
HELD_OUT_INTERVENTION: tuple[InterventionSpec, ...] = (
    InterventionSpec("TidyUp-v1", "resource_contention", "held_out_intervention"),
    InterventionSpec("TidyUp-v1", "resource_contention_temporary", "held_out_intervention"),
)

INTERVENTION_SPLITS: dict[str, tuple[InterventionSpec, ...]] = {
    "train": INTERVENTION_TRAIN,
    "held_out_intervention": HELD_OUT_INTERVENTION,
}


def all_intervention_specs() -> tuple[InterventionSpec, ...]:
    """Every intervention spec across every split, in split order (train
    first) -- same shape as all_specs() above, for the intervention axis."""
    return INTERVENTION_TRAIN + HELD_OUT_INTERVENTION
