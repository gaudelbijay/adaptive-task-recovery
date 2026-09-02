# Review request: task schema and everything built on it

**Status: RESOLVED 2026-08-02 (D-037) — self-resolved, not independently
reviewed.** The project owner directed resolving this without waiting
further for the teammate this document was addressed to, rather than
leaving the project blocked. All four questions below got a real answer
(one led to an actual fix, not just a decision), and the schema was
promoted to `src/atr/` on that basis. **This is not the same thing as
independent review** — nobody who didn't already believe the schema was
right evaluated it. If the actual teammate reviews this later and
disagrees with any call made here, that's a real reopening, not just a
formality; nothing below should be read as carrying more confidence than
"the project owner and I worked through it," which is a different, weaker
claim than "a second person with their own judgment checked it." Original
request preserved below for the historical record of what was asked and
why.

---

**For:** the teammate on this project (representation, language,
feasibility side)
**Ask:** review D-013's task schema and decide whether it's ready to move
from `spikes/task_schema_draft/` into `src/atr/` as committed architecture,
or needs changes first.
**Why now:** D-013 has said "needs review with teammate before Accepted"
since 2026-07-29. Everything since (D-014 through D-029) is built on top of
that unreviewed schema — all six build-up stages now pass a combined test
suite (97 tests), including a real end-to-end pass (D-029) that combines
language parsing, real vision-based feasibility, and a learned policy in
one episode with nothing privileged in the live decision loop. This is
about as far as this line of work can reasonably go without the review —
a natural point to actually get it rather than keep building further on an
unconfirmed foundation.

## What's being proposed

*(Now promoted — see D-037. Paths below are where this code lived when
this document was first written; it now lives at `src/atr/language/
goal_graph.py`, `src/atr/feasibility/oracle.py`, and
`src/atr/constraints/intent_guard.py` respectively.)*

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
- **Language**: a controlled-grammar parser (`instruction_parser.py`, D-019)  turns
  an instruction sentence into a `GoalGraph`, instead of one being
  hand-written — reproduces all existing hand-authored graphs from their
  own text and generalizes to held-out paraphrases and a held-out object
  composition. Extended (D-026) to handle ordering/priority ("first X,
  then Y") and conditional goals ("if X is destroyed, do Y instead").
- **Vision**: zero-shot CLIP (`clip_feasibility.py`, D-020) judges object presence
  from a rendered frame instead of privileged state, matching oracle
  feasibility on the cases tested. Now validated on a second, independently
  calibrated scene layout too (D-027), not just one.
- **Self-supervised representation**: a DINOv2 (no text/label supervision)
  linear probe (`dinov2_probe.py`, D-023) separates object-present from
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
- **Everything combined, D-029**: `end_to_end.py` runs one real episode
  where the instruction is parsed, feasibility comes from a rendered frame
  (not privileged state), and a trained Q-table (not a hard-coded rule)
  decides what to attempt — matches oracle exactly. This is the last stage
  in the build-up order this schema has been developed against; there
  isn't a natural next increment to build without your review first.

Full narrative, in order, with what broke and how it got fixed at each
step: `spikes/task_schema_draft/README.md`. Full rationale for each
decision: `ai-notes/decisions.md`, D-013 through D-029.

## Specific questions worth your judgment — RESOLVED (D-037, 2026-08-02)

1. **Is the goal/constraint shape actually right?** — **Accepted as-is,
   no changes.** Two predicate/constraint kinds exist (`on_tray`,
   `never_move`, `maintain_orientation`) because that's all any worked
   example so far has needed. `Literal` types plus `constraint_violated()`
   raising `ValueError` on an unrecognized kind (fails loudly, never
   silently) make adding a new kind later a contained, safe change —
   nothing here blocks extension when a real driving case shows up. Not
   extended speculatively now, matching this project's discipline
   elsewhere (see the "Preferences" gap noted below, deliberately left
   unaddressed the same way).
2. **Is `Goal.condition` (D-026) the right shape for conditional goals?**
   — **Accepted as-is, kept scoped to object existence.** Deliberately
   *not* extended to reference another goal's feasibility — that's now
   `Goal.depends_on`'s job (question 3, resolved below), which makes
   `condition` and `depends_on` two distinct, complementary mechanisms
   instead of one field trying to do both: `condition` gates on an
   *object's* existence, `depends_on` gates on a *goal's* completion.
   Still a single condition, not a list — no instruction pattern parsed
   so far needs more than one, and adding that without a driving case
   would be the same mistake question 1 declined to make.
3. **`Goal.depends_on` is still an unexercised schema field.** — **Fixed,
   not just decided.** It really was dead schema surface: defined since
   D-013's first draft, read by zero functions. Built
   `goal_dependencies_satisfied()` (`src/atr/feasibility/oracle.py`) — a
   hard prerequisite gate, deliberately a *separate* function from
   `goal_feasible()` rather than folded into it, since "infeasible"
   (can never be achieved) and "dependency not yet satisfied" (would
   succeed later) are genuinely different claims that a policy needs to
   tell apart. Added `dependent_goals_example()`
   (`src/atr/language/goal_graph.py`) and wired the gate into
   `feasibility_aware_policy()`
   (`spikes/task_schema_draft/policy_baselines.py`) — verified with both
   pure-function tests (`tests/drafts/test_oracle_feasibility.py::
   TestGoalDependency`) and a real live-env demonstration
   (`tests/drafts/test_policy_baselines.py::TestGoalDependencyGating`):
   `place_bowl` (depends on `place_mug`) gets blocked when `red_mug` is
   destroyed, even though `place_bowl`'s own target object (`blue_bowl`)
   is entirely untouched and independently feasible — the dependency,
   not feasibility, is what's stopping it. Along the way, also confirmed
   `Goal.priority` is currently *set* by `instruction_parser.py` but read
   by nothing — harmless (goal execution order already matches tuple
   order, which priority is derived from), but worth knowing if anything
   ever assumes priority is independently load-bearing; it isn't yet.
4. **Is toy-scale, single-instruction, two-scene evidence enough to
   promote this to `src/`?** — **Yes, promoted.** The accumulated case:
   six build-up stages, four robot/scene combinations for the policy
   layer, two independently-calibrated scene layouts for vision, 121
   tests (97 before this resolution + the new dependency tests), and a
   real end-to-end pipeline with nothing privileged in the live decision
   loop (D-029). Promotion changes *where the code lives and its status*
   (Proposed → Accepted) — it does not retroactively make the underlying
   evidence less toy-scale. Every caveat below still applies exactly as
   written; promoting doesn't erase them.

## Known caveats — not hidden, worth reading before deciding

- **Everything is toy-scale.** One canonical instruction (plus its
  ReplicaCAD/humanoid variants), a handful of objects, small sample sizes
  throughout (e.g. dinov2_probe.py's probe: grown to 20 examples, still
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
- **`Goal.condition` (D-026) is no longer unreviewed** — resolved as
  question 2 above (accepted as-is, kept scoped to object existence).

## How to look at this yourself

```
cd spikes/task_schema_draft
cat README.md                      # full narrative, in order
python -m pytest ../../tests/ -q   # ~121 tests, ~7-8 minutes on this machine
```

Needs the `.maniskill` pyenv virtualenv
(`requirements-maniskill.lock.txt`) — see `ai-notes/decisions.md` D-009 for
setup notes if you're on a different machine.

## Also worth knowing before you dig in

`spikes/task_schema_draft/` also just went through a professional file/
function renaming pass (D-030) — `language.py` → `instruction_parser.py`,
`vision.py` → `clip_feasibility.py`, `representation.py` →
`dinov2_probe.py`, and a `train_q_table_for_replicacad_humanoid()` /
`train_q_policy()` duplication collapsed into one shared `train_q_table()`.
No behavior change, full suite re-verified green after it — flagging only
so file names in this document match what you'll actually see if you
pulled an older mental model of the layout.

`ai-notes/status.md`, `todo.md`, and `recent_changes.md` were stale
(untouched since 2026-07-26) as of the first draft of this document; they've
since been consolidated into short stubs pointing at the root-level
`status.md`, which is what's actually kept current. Nothing left to
reconcile there.
