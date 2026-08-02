# Review request: task schema and everything built on it

**For:** the teammate on this project (Person A / representation, language,
feasibility side)
**Ask:** review D-013's task schema and decide whether it's ready to move
from `spikes/task_schema_draft/` into `src/atr/` as committed architecture,
or needs changes first.
**Why now:** D-013 has said "needs review with teammate before Accepted"
since 2026-07-29. Everything since (D-014 through D-028) is built on top of
that unreviewed schema — five separate build-up stages now pass a combined
test suite, which makes this a natural point to actually get the review
rather than keep building further on an unconfirmed foundation.

## What's being proposed

`spikes/task_schema_draft/goal_graph.py` defines the schema docs/04 asked
for: a `Goal` (id, predicate, target object, priority, dependencies), a
`Constraint` (id, kind — `never_move` or `maintain_orientation` — target
object, tolerance), and a `GoalGraph` (instruction text, goals, constraints)
tying them to a language instruction. `oracle_feasibility.py` is a pure,
simulator-independent implementation of feasibility and constraint-violation
checking against that schema. `intent_guard.py` blocks actions that would
violate a constraint unless a real goal requires them.

That's the entire proposal, plus one addition since the first draft of
this document: `Goal.condition: tuple[str, bool] | None` (D-026) — gates
whether a goal is "in play" at all this episode, for "if X is destroyed,
do Y instead" instructions. Flagged the same way D-013's original fields
are — PROPOSED, not Accepted, needing exactly this review.

Everything else in `spikes/task_schema_draft/` is evidence for or against
the schema, not part of it.

## What's been validated against it

- **H2 (feasibility-aware beats static)** and **H3 (intent guard blocks
  violations at low recall cost)** both have first, toy-scale, runnable
  tests (D-014, D-015) — not just stated as hypotheses anymore.
- **Embodiment-agnostic, checked on four different robot/scene
  combinations**, not assumed: Panda arm on a tabletop (original), a
  Unitree G1 humanoid on a kitchen counter (D-016), a real ReplicaCAD
  apartment with a Fetch mobile robot doing real navigation (D-017), and
  G1 placed in that same real apartment (D-018). Same H2/H3 results all
  four times, once real placement/navigation bugs were found and fixed
  (D-018, D-021).
- **Language**: a controlled-grammar parser (`language.py`, D-019)  turns
  an instruction sentence into a `GoalGraph`, instead of one being
  hand-written — reproduces all existing hand-authored graphs from their
  own text and generalizes to held-out paraphrases and a held-out object
  composition. Extended (D-026) to handle ordering/priority ("first X,
  then Y") and conditional goals ("if X is destroyed, do Y instead").
- **Vision**: zero-shot CLIP (`vision.py`, D-020) judges object presence
  from a rendered frame instead of privileged state, matching oracle
  feasibility on the cases tested. Now validated on a second, independently
  calibrated scene layout too (D-027), not just one.
- **Self-supervised representation**: a DINOv2 (no text/label supervision)
  linear probe (`representation.py`, D-023) separates object-present from
  object-absent at least as well as CLIP did, on the same task. Grown to a
  20-example headline result and also supports the second scene layout.
- **Learned policy**: tabular Q-learning (`rl_policy.py`, D-025), trained
  on real environment rollouts, discovers "attempt iff feasible" from
  reward alone and matches the hand-coded rule exactly.
- **Real IK, tried and honestly reported as insufficient**: D-024/D-028
  built a proper analytic-Jacobian IK solver (`ik_solver.py`, on
  `pinocchio`, verified against ManiSkill's own kinematics) to attempt real
  contact-based grasp confirmation. Confirmed, not guessed: neither target
  object is within real reach of G1's arm from any reasonable standing
  position in this scene. Included here because a review of "what's been
  validated" should include a well-investigated negative result, not just
  the wins.

Full narrative, in order, with what broke and how it got fixed at each
step: `spikes/task_schema_draft/README.md`. Full rationale for each
decision: `ai-notes/decisions.md`, D-013 through D-028.

## Specific questions worth your judgment

1. **Is the goal/constraint shape actually right?** Two predicate/constraint
   kinds exist (`on_tray`, `never_move`, `maintain_orientation`) because
   that's all any worked example so far has needed. Does your side of the
   work (representation/feasibility) need anything the schema doesn't
   currently express?
2. **Is `Goal.condition` (D-026) the right shape for conditional goals?**
   It's a single (object_id, required_exists) pair checked against
   privileged state — the simplest thing that could support "if X is
   destroyed, do Y instead". Does it need to reference another *goal's*
   feasibility instead of an object's existence directly, or support more
   than one condition, or something else entirely? This is the one piece
   of schema surface added since the first draft of this document, and
   it's the one most worth your scrutiny specifically.
3. **`Goal.depends_on` is still an unexercised schema field.**
   `Goal.priority` now has a real example (D-026); `depends_on` (ordering
   dependencies between goals, not scoring priority) still doesn't. Worth
   a real example that uses it before trusting the field is shaped right?
4. **Is toy-scale, single-instruction, two-scene evidence enough to
   promote this to `src/`?** Or does moving it there imply a confidence
   level none of this actually supports yet?

## Known caveats — not hidden, worth reading before deciding

- **Everything is toy-scale.** One canonical instruction (plus its
  ReplicaCAD/humanoid variants), a handful of objects, small sample sizes
  throughout (e.g. representation.py's probe: grown to 20 examples, still
  not a statistical claim). See each decision's own "Consequences" section.
- **D-022: a confirmed, open, unfixed upstream ManiSkill3 rendering bug**
  (haosulab/ManiSkill#1150) limits how many rendered frames can be trusted
  per process for the real-scene envs. Guarded with a runtime warning, not
  fixed — genuinely can't be fixed here (it's in a dependency, not this
  project's code).
- **D-024/D-028: real contact/tactile grasp confirmation was attempted
  twice and confirmed infeasible**, not just "not yet working." The second
  attempt used a proper analytic-Jacobian IK solver (verified against
  ManiSkill's own kinematics, fully deterministic) with a wide
  random-restart search across candidate base positions — neither target
  object comes within real contact range of G1's arm from any reasonable
  standing position in this scene. `teleport-on-success` remains the
  manipulation abstraction everywhere — no existing result depends on real
  grasp precision, but this is a real, now well-confirmed gap, not a
  hidden one.
- **The vision/representation work is calibrated to two specific scene
  layouts, not a distribution over layouts** (D-021 pinned scene layout
  deliberately, for reasons specific to G1's placement; D-027 added a
  second one specifically so this wasn't validated on only one) — still
  not a generalization test in any statistical sense.
- **`Goal.condition` (D-026) is new schema surface, added since this
  document was first written, and is explicitly not yet reviewed** — see
  question 2 above.

## How to look at this yourself

```
cd spikes/task_schema_draft
cat README.md                      # full narrative, in order
python -m pytest ../../tests/ -q   # ~95 tests, ~5 minutes on this machine
```

Needs the `.maniskill` pyenv virtualenv
(`requirements-maniskill.lock.txt`) — see `ai-notes/decisions.md` D-009 for
setup notes if you're on a different machine.

## Also worth knowing before you dig in

`ai-notes/status.md`, `todo.md`, and `recent_changes.md` have not been
touched since 2026-07-26 and do not reflect any of D-013 through D-025 —
the root-level `STATUS.md` is what's actually been kept current. Worth
reconciling those, separately from this review.
