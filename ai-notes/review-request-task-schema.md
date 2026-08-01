# Review request: task schema and everything built on it

**For:** the teammate on this project (Person A / representation, language,
feasibility side)
**Ask:** review D-013's task schema and decide whether it's ready to move
from `spikes/task_schema_draft/` into `src/atr/` as committed architecture,
or needs changes first.
**Why now:** D-013 has said "needs review with teammate before Accepted"
since 2026-07-29. Everything since (D-014 through D-025) is built on top of
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

That's the entire proposal. Everything else in `spikes/task_schema_draft/`
is evidence for or against it, not part of the schema itself.

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
- **Language**: a controlled-grammar parser (`language.py`, D-019) turns
  an instruction sentence into a `GoalGraph`, instead of one being
  hand-written — reproduces all existing hand-authored graphs from their
  own text and generalizes to held-out paraphrases and a held-out object
  composition.
- **Vision**: zero-shot CLIP (`vision.py`, D-020) judges object presence
  from a rendered frame instead of privileged state, matching oracle
  feasibility on the cases tested.
- **Self-supervised representation**: a DINOv2 (no text/label supervision)
  linear probe (`representation.py`, D-023) separates object-present from
  object-absent at least as well as CLIP did, on the same task.
- **Learned policy**: tabular Q-learning (`rl_policy.py`, D-025), trained
  on real environment rollouts, discovers "attempt iff feasible" from
  reward alone and matches the hand-coded rule exactly.

Full narrative, in order, with what broke and how it got fixed at each
step: `spikes/task_schema_draft/README.md`. Full rationale for each
decision: `ai-notes/decisions.md`, D-013 through D-025.

## Specific questions worth your judgment

1. **Is the goal/constraint shape actually right?** Two predicate/constraint
   kinds exist (`on_tray`, `never_move`, `maintain_orientation`) because
   that's all any worked example so far has needed. Does your side of the
   work (representation/feasibility) need anything the schema doesn't
   currently express?
2. **Priorities and dependencies are schema fields nobody has exercised.**
   `Goal.priority` and `Goal.depends_on` exist per docs/04's requirement but
   every example so far has equal-priority, independent goals. Worth a real
   example that uses them before trusting the fields are shaped right?
3. **Is toy-scale, single-instruction, single-scene evidence enough to
   promote this to `src/`?** Or does moving it there imply a confidence
   level none of this actually supports yet?

## Known caveats — not hidden, worth reading before deciding

- **Everything is toy-scale.** One canonical instruction (plus its
  ReplicaCAD/humanoid variants), a handful of objects, small sample sizes
  throughout (e.g. representation.py's probe: 8 examples). None of this is
  a statistical claim — see each decision's own "Consequences" section.
- **Ordering/priority and conditional goals are unimplemented**, not just
  untested — `language.py` has no grammar for them (D-019).
- **D-022: a confirmed, open, unfixed upstream ManiSkill3 rendering bug**
  (haosulab/ManiSkill#1150) limits how many rendered frames can be trusted
  per process for the real-scene envs. Guarded with a runtime warning, not
  fixed — can't be fixed here.
- **D-024: real contact/tactile grasp confirmation was attempted and found
  infeasible** with current tooling (G1's reach can't get close enough to
  the object for real contact; a closed-loop IK solver built to fix this
  converged unreliably). `teleport-on-success` remains the manipulation
  abstraction everywhere — no existing result depends on real grasp
  precision, but this is a real, acknowledged gap, not a hidden one.
- **The vision/representation work is calibrated to one specific scene
  layout** (D-021 pinned it deliberately, for reasons specific to G1's
  placement) — not a generalization test.

## How to look at this yourself

```
cd spikes/task_schema_draft
cat README.md                      # full narrative, in order
python -m pytest ../../tests/ -q   # 79 tests, ~4 minutes on this machine
```

Needs the `.maniskill` pyenv virtualenv
(`requirements-maniskill.lock.txt`) — see `ai-notes/decisions.md` D-009 for
setup notes if you're on a different machine.

## Also worth knowing before you dig in

`ai-notes/status.md`, `todo.md`, and `recent_changes.md` have not been
touched since 2026-07-26 and do not reflect any of D-013 through D-025 —
the root-level `STATUS.md` is what's actually been kept current. Worth
reconciling those, separately from this review.
