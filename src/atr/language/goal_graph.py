"""Goal graph data structures matching docs/04-benchmark-environment.md's
task schema: "a language instruction compiled into atomic goals, priorities,
dependencies, and hard constraints."

This is a DRAFT for the "Shared: select the task family and
irreversible/reversible intervention set" item in ai-notes/status.md — a concrete
starting point for review, not a committed schema. See ../README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Goal:
    """One atomic, checkable goal — e.g. "place the red mug on the tray"."""

    id: str
    predicate: Literal["on_tray"]
    target_object: str
    priority: int = 0
    depends_on: tuple[str, ...] = ()
    # PROPOSED extension (D-026, ai-notes/decisions.md), not yet reviewed --
    # same "needs review" status as the rest of this schema (D-013). Gates
    # whether this goal is "in play" at all this episode: (object_id,
    # required_exists). goal_feasible() treats the goal as infeasible
    # outright if state[object_id].exists != required_exists, before even
    # checking the goal's own target_object. Lets an instruction express
    # "if X is destroyed, do Y instead" -- see instruction_parser.py's
    # conditional clause grammar and oracle_feasibility.py's goal_feasible().
    condition: tuple[str, bool] | None = None


@dataclass(frozen=True)
class Constraint:
    """A hard constraint that must hold throughout the episode, independent
    of goal completion — violating one is never acceptable for reward."""

    id: str
    kind: Literal["never_move", "maintain_orientation"]
    target_object: str
    # position tolerance in meters (never_move) or minimum "up-vector" alignment,
    # i.e. cos(max tilt angle) (maintain_orientation)
    tolerance: float = 0.02


@dataclass(frozen=True)
class GoalGraph:
    """Everything docs/04's task schema asks the instruction to compile into."""

    instruction_text: str
    goals: tuple[Goal, ...]
    constraints: tuple[Constraint, ...]


CANONICAL_INSTRUCTION_TEXT = (
    "Put the red mug and blue bowl on the tray, keep the medicine "
    "upright, and do not move the glass."
)
CANONICAL_OBJECTS = frozenset({"red_mug", "blue_bowl", "medicine_bottle", "glass"})


def dependent_goals_example() -> GoalGraph:
    """Exercises `Goal.depends_on` -- part of D-013's schema from the
    start, but never read by any function until `oracle_feasibility.py`'s
    `goal_dependencies_satisfied()` (D-037; see
    ai-notes/review-request-task-schema.md question 3). Reuses
    canonical_example()'s real objects (red_mug, blue_bowl) so it runs
    against the same tidy_up_env.py scene: an artificial but genuine hard
    ordering, place_bowl can't be attempted until place_mug is actually
    *achieved* -- not just feasible, and not just earlier by tuple order
    (every goal already is that, by construction; depends_on is a
    strictly stronger guarantee than Goal.priority, which no code
    currently reads either -- see goal_dependencies_satisfied()'s
    docstring)."""
    return GoalGraph(
        instruction_text="Put the red mug on the tray, then the blue bowl.",
        goals=(
            Goal(id="place_mug", predicate="on_tray", target_object="red_mug", priority=0),
            Goal(
                id="place_bowl", predicate="on_tray", target_object="blue_bowl",
                priority=1, depends_on=("place_mug",),
            ),
        ),
        constraints=(),
    )


def canonical_example() -> GoalGraph:
    """The project's own worked example — docs/01-problem-statement-and-motivation.md
    "Example": "Put the red mug and blue bowl on the tray, keep the medicine
    upright, and do not move the glass." If the bowl irreversibly breaks, a
    valid agent infers the bowl goal is infeasible, still places the mug,
    and never moves the glass merely because it offers an easier route.

    Hand-authored: this is the reference instruction_parser.py's parser is
    checked against (see tests/drafts/test_instruction_parser.py), not
    itself produced by parsing. tidy_up_env.py uses
    parse_instruction(CANONICAL_INSTRUCTION_TEXT, CANONICAL_OBJECTS)
    instead of this function, for a real one."""
    return GoalGraph(
        instruction_text=CANONICAL_INSTRUCTION_TEXT,
        goals=(
            Goal(id="place_mug", predicate="on_tray", target_object="red_mug", priority=0),
            Goal(id="place_bowl", predicate="on_tray", target_object="blue_bowl", priority=0),
        ),
        constraints=(
            Constraint(
                id="keep_medicine_upright", kind="maintain_orientation",
                target_object="medicine_bottle", tolerance=0.85,
            ),
            Constraint(
                id="dont_move_glass", kind="never_move",
                target_object="glass", tolerance=0.02,
            ),
        ),
    )
