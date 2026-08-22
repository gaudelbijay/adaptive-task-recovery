---
title: Problem Statement and Motivation
status: draft
last_updated: 2026-08-01
---

# Problem Statement and Motivation

## The problem

At episode start, an embodied agent receives a natural-language instruction with
multiple desired outcomes and possibly hard constraints. During execution, the
world changes in a way that is unforeseen, persistent, and not practically
reversible. The original plan is no longer valid; some goals may remain feasible,
some may require a different strategy, and others may be impossible.

The agent must infer that new feasibility structure from visual observations and
history, then act to maximize legitimate goal achievement while respecting the
instruction's hard constraints and semantic content.

## Key distinctions

- **Difficulty is not infeasibility:** a blocked direct route may admit a detour;
  a destroyed required object may make its associated goal impossible.
- **Adaptation is not restoration:** the system is not expected to return the
  world to its previous state.
- **Partial completion is not arbitrary reward maximization:** remaining goals
  may be completed only if doing so respects dependencies and constraints.
- **Intent preservation is operational, not philosophical:** v1 measures
  compliance with explicit predicates, priorities, and equivalence rules in the
  benchmark. It does not claim to solve general intent alignment.

## Example

Instruction: “Put the red mug and blue bowl on the tray, keep the medicine
upright, and do not move the glass.” If the bowl irreversibly breaks, a valid
agent should infer that the bowl goal is infeasible, still place the mug if that
does not violate another constraint, and never move the glass merely because it
offers an easier route. An invalid agent might loop on the bowl, substitute an
unrequested object, or violate the glass constraint for reward.

## Research hypotheses

- **H1 — representation:** self-supervised visual representations improve
  feasibility prediction and held-out-change generalization over pixels trained
  only through task reward and standard supervised features. **First toy-scale
  test (2026-08-01):** see D-023 in `ai-notes/decisions.md` and
  `spikes/task_schema_draft/dinov2_probe.py` — a DINOv2 (self-supervised,
  no text/label supervision) linear probe separates object-present from
  object-absent at least as well as zero-shot CLIP did (D-020) on the same
  task. Not a comparison against "pixels trained only through task reward,"
  which doesn't exist yet, and not held-out-change generalization (both
  models were only tested on the same object/scene they were calibrated
  against) — an existence proof that the representation *can* support this
  judgment, not a test of the comparative claim H1 actually makes. **First
  live-loop test (2026-08-04):** see D-054 in `ai-notes/decisions.md` — wiring
  the DINOv2 probe into an actual decision loop (not just LOO evaluation on
  calibration captures) surfaced a real robustness gap the earlier existence
  proof couldn't: on a live episode's second goal, G1's arm has already moved
  from the first goal's attempt, producing a frame never seen during
  training (all calibration captures are arm-at-rest); the probe confidently
  (81%) misjudged a genuinely destroyed object as present, while CLIP's
  zero-shot judgment on the identical frame was correct. At the time, this
  cut against a naive reading of H1 — evidence that this self-supervised
  probe, calibrated only on arm-at-rest data, generalized *worse* than the
  language-supervised baseline to a realistic distribution shift.
  **Root-caused and closed (2026-08-04, D-055):** the gap traced to training
  data, not a representational ceiling — a probe fit on arm-at-rest examples
  *plus* examples from the same post-first-attempt state the live loop's
  second goal actually renders (arm moved, first object teleported into the
  tray) matched oracle on the original failing case and 4 further held-out
  seeds/conditions. So the fuller picture as of D-055: this self-supervised
  representation *can* support a robust decision under a realistic
  distribution shift, but — unlike CLIP's zero-shot judgment, which needed
  no shift-specific data at all — only once the training data actually
  covers that shift. That gap in what each approach needs to generalize is
  itself relevant evidence for H1, not fully for the self-supervised side
  and not fully against it. **Roles formalized 2026-08-06 (D-062, closing
  I-004):** DINOv2 is the project's committed self-supervised baseline for
  this comparison; CLIP is kept permanently as the required language-
  supervised reference point, not a competing "selection" to eliminate —
  H1's own claim can't be tested without both. **The missing comparison
  point built and measured (2026-08-06, D-066):** the original test above
  explicitly flagged "pixels trained only through task reward" as not
  existing yet — it now does
  (`src/atr/feasibility/task_reward_encoder.py`): a small conv encoder,
  no pretraining of any kind, trained from scratch on the identical
  toy-scale data (same object, same scene, same 6-present/6-absent LOO
  setup) CLIP and DINOv2 were both evaluated against. Result, root-caused
  before trusting it: 0% LOO accuracy, and not from noisy guessing —
  every held-out example in every fold got the exact same output
  regardless of image content, confirmed directly (near-zero logit
  variance across all 12 images, in every fold), meaning the model
  never learned to look at the image at all; it just predicted whichever
  class happened to be the majority in that fold's own training split.
  Real gradient flow and real weight changes were confirmed too, ruling
  out a training bug rather than assuming the result. This is the
  clearest, most direct evidence for H1's actual comparative claim in
  the project so far: given the exact same tiny amount of task data,
  both pretrained representations (CLIP's zero-shot judgment, needing no
  training data at all, and DINOv2's self-supervised pretraining plus a
  fitted probe) reach 100% LOO accuracy, while training visual features
  from scratch on that same data doesn't learn to discriminate at all.
  Still bounded: toy-scale, one object/scene, and a reward-*derived*
  supervised loss standing in for literal online policy-gradient RL
  (disclosed, not hidden) — not a claim that no amount of task-reward
  training could ever work, only that it doesn't at this project's
  current data scale, in contrast to the pretrained alternatives which
  succeed at that same scale. **CLIP's own "needs no shift-specific data"
  framing above needed a correction, 2026-08-10 (D-088):** running the
  full live-agent pipeline through a real multi-seed benchmark for the
  first time (not just the single hand-checked episode D-054 used)
  surfaced that CLIP has an analogous robustness gap of its own, in a
  different env/scene (ReplicaCAD-Humanoid, `master_chef_can`,
  `kitchen_cabinet`) and the opposite direction (a false negative, not a
  false positive): once G1's arm occupies the calibrated crop region after
  a real prior goal attempt, CLIP judged the object "absent" in every one
  of 8 episodes tested, including all 7 where it was genuinely still
  present. Visually confirmed, not just measured. D-054's specific
  finding (CLIP correct where DINOv2 wasn't, on *that* frame, in the
  canonical env) still holds — but "CLIP needs no shift-specific data"
  was never actually tested against *this* shift (post-attempt arm
  position in a different env/scene) before D-088, and doesn't hold up
  once it is. Both representations tested so far turn out to have
  real, distinct robustness gaps under a live decision loop's actual
  rendered states — a more precise, and more interesting, H1-relevant
  picture than either one being simply "robust." **CLIP's gap fixed,
  2026-08-10 (D-089):** unlike DINOv2's fix (D-055, add representative
  training data), CLIP is zero-shot — fixed by recalibrating the crop
  geometry itself (prompt unchanged), found by measuring several
  candidates directly against real present/absent frames rather than
  guessing. 0/8 mismatches after the fix (down from 7/8), and the
  pre-existing arm-at-rest calibration test still passes unchanged — the
  fix generalizes across both visual states. Both representations tested
  in this project have now had a real live-loop robustness gap found
  *and* fixed (D-055 for DINOv2, D-089 for CLIP) — a genuinely symmetric,
  non-cherry-picked picture of what it actually takes to make either one
  reliable in a live decision loop, not just at calibration time.
- **H2 — explicit feasibility:** conditioning strategy selection on per-goal
  feasibility estimates outperforms a static language-conditioned policy after
  irreversible changes. **First toy-scale test (2026-07-29):** see D-014 in
  `ai-notes/decisions.md` and `spikes/task_schema_draft/policy_baselines.py`
  — a hand-authored single scenario, not evidence for the general claim, but
  the first time this hypothesis has been run rather than just stated.
  Since then: confirmed with the same result across four robot/scene
  combinations (D-016–D-018, D-021), and — closer to what H2 actually asks —
  a tabular Q-learning policy (D-025) *discovers* "condition on feasibility"
  from reward alone, rather than having it hand-coded, and matches the
  hard-coded policy's behavior exactly. Still toy-scale (privileged-state
  feasibility, 2-goal instructions); still not the general claim.
  **A real limitation of the "condition on feasibility" strategy itself
  surfaced 2026-08-07 (D-070), not sought — found investigating an
  unexpected result while giving the evaluation harness genuine timing
  variance to measure:** once intervention timing is wide enough to span a
  goal's own ~25-step attempt duration (not the narrow windows every
  earlier H2 comparison used), "currently feasible" stops reliably
  predicting "will complete" -- measured directly, 72.5% (29/40) of cases
  perceived feasible at decision time were destroyed mid-attempt anyway,
  *conditional on the risky intervention actually being active*. That
  conditional mechanism is real and held up under further scrutiny (D-071,
  bootstrap CI clearly excludes zero, `n=198`). **D-070's further claim —
  that a reward-trained Q-learning policy "correctly (not buggily) learns
  to skip it," as the mathematically optimal response — was an overclaim,
  corrected 2026-08-08 (D-071):** the Q-table's negative value was trained
  on a *pooled* state key (`(goal_id, feasible)`, averaged across both the
  risky intervention and episodes with no risk at all), and the true
  expected value of that pooled quantity is statistically indistinguishable
  from zero (CI straddles zero, `n=441`) — the confidently-negative Q-value
  was a small-sample TD-learning artifact, not a genuine discovery, shown
  directly by its instability under more training. Building an explicit
  calibration primitive that keys on `(goal_id, intervention_kind)` instead
  of pooling (`src/atr/feasibility/calibrated_feasibility.py`, D-071)
  recovers the decisive conditional answer directly and adapts correctly
  when no risk is present, something `feasibility_aware_policy`'s hard-coded
  binary rule and the pooled-state Q-table both cannot express. The deeper,
  still-standing implication for H2: a binary existence check is an
  incomplete feasibility signal once mid-attempt risk is real, and a state
  representation that pools across *why* a goal might currently be at risk
  is not enough either — a calibrated probability keyed on the actual
  mechanism was needed here, closer to H5's calibration question than H2's
  original framing. Whether attempting is worth the risk once the true
  conditional trade-off is properly captured — the ~18%-of-cases upside
  D-070 originally described — is real and matches
  `feasibility_aware_policy`'s own measured behavior; a fully reward-
  optimal policy needs both the calibration *and* the mechanism-aware
  state, not either alone.
- **H3 — intent guard:** explicit goal/constraint checking reduces semantic and
  constraint violations with an acceptable trade-off in achievable-goal recall.
  **First toy-scale test (2026-07-29):** see D-015 and
  `src/atr/constraints/intent_guard.py` (promoted from `spikes/task_schema_draft/`,
  D-037) — blocks one hand-authored
  constraint violation at zero recall cost; this did not test the harder
  recall/safety trade-off the hypothesis is actually about (see R-010).
  Confirmed with the same result across the same four robot/scene
  combinations as H2 (D-016–D-018, D-021). **Trade-off quantified
  2026-08-09 (D-058/D-082):** on five independently labelled candidates,
  including direct goal/constraint tension and conditional fallback before
  and after activation, the state-aware guard has legitimate-action recall
  1.0 and unsafe-action violation rate 0.0. Removing state keeps recall 1.0
  but raises violation rate to 0.5. This closes the constructible high-level
  case. **Predicted side effects represented (2026-08-09, D-083):** actions may
  now carry an `affected_objects` set, so reaching for a legitimate target is
  blocked when a motion/skill predictor says it would disturb a protected
  object. The guard consumes those predictions; it does not yet infer physical
  contacts itself. **A real predictor built and wired in (2026-08-09,
  D-084–D-087):** `constraints/effect_predictor.py` supplies the
  `affected_objects` set D-083's interface needed — every existing object
  within a conservative swept corridor around a planned motion, excluding the
  intended target (already handled as an implicit effect). Extended to check
  every segment of a real bent multi-waypoint path, not a straight start-end
  chord (D-085), and to account for each object's own extent/collision
  radius, not just its center point (D-086) — both closing real, disclosed
  false-negative gaps in the geometry, not just adding features. Wired into
  the first real planner interface the safety layer actually consumes
  (D-087, `screen_navigation_path()`, `envs/navigation.py`, on the real
  Fetch navigation stack): for the same legitimate red-mug target, a route
  whose later leg passes the protected glass is blocked, and a clear detour
  is allowed. This closes R-010's long-standing "not fully closed" caveat
  (the physical-obstruction scenario is now representable and tested
  end-to-end) but the model is still conservative sphere/point-based
  geometry, not real robot-link collision checking, and blocked-route
  behavior (stop vs. replan) isn't yet wired into `_navigate_to()`'s actual
  execution contract — a real, disclosed remaining gap, not a finished
  claim. **That remaining gap closed, 2026-08-12/13 (D-091–D-100):**
  `_navigate_to()` screens the real planned route and stops safely (zero
  motion) when it's rejected (D-091), then D-092 adds a constraint-aware
  detour search — inflate the predicted hazard into the cached occupancy
  grid, replan, re-screen — before falling back to that stop. Validated
  with fully live Fetch execution, no mocks: a real positive-detour episode
  (D-096, 250 real steps, goal achieved, protected object displaced
  exactly `0.0 m`); a direct stop-only-vs-replan safety-matched-recall
  comparison answering R-010's original "safe by doing nothing" concern
  (D-097 — stop-only preserves the object but skips an achievable goal;
  replanning keeps the same zero displacement *and* completes it); the
  same positive result across three hazard locations and both goal routes
  (D-098/D-099, six controlled cases); and confirmation the behavior
  follows `GoalGraph` constraint semantics rather than a hardcoded object
  name, via a second protected-object type (D-100). Every live case so far
  is one scene/layout/seed, not a distribution of naturally occurring
  hazards — the honest remaining scope, not assumed broader.
- **H4 — compositional generalization:** factorized goal and change
  representations transfer better to unseen goal-change combinations than a
  monolithic policy. **First real comparative test (2026-08-09, D-079):**
  investigated the intervention-mechanism axis first and found a real
  scoping limit — every intervention kind in every env variant threatens
  exactly one specific goal, so no env has genuine multi-goal ×
  multi-intervention structure to hold a combination out from. Tested the
  language axis instead, where that structure already exists
  (`atr.evaluation.splits`'s `held_out_composition` spec, D-044): built a
  monolithic exact-string-memorization baseline
  (`src/atr/language/compositional_generalization.py`) and compared it
  against the real, factorized `instruction_parser.py` across train,
  held-out-paraphrase, and held-out-composition. Result: the factorized
  parser is 100% correct on every split (1/1, 3/3, 1/1); the monolithic
  baseline is 100% on train (exactly what it memorized) and 0% on both
  held-out splits. A real, decisive first data point, honestly scoped —
  the parser's factorization is a hand-written grammar, not a learned
  representation, and the monolithic baseline is a maximally weak
  strawman (zero generalization mechanism, not just a weaker learned one);
  a genuinely learned monolithic baseline might do better on paraphrases
  specifically. **Stronger surface baseline (2026-08-09, D-080):** a trained
  character-trigram nearest-neighbor retriever removes that easy confound: it
  transfers its indivisible training graph across every unseen paraphrase
  (3/3) but still fails the novel composition (0/1), where the factorized
  parser succeeds (1/1). This isolates composition from mere string novelty,
  while remaining a tiny controlled-language result rather than evidence from
  a neural sequence model or simulator-level goal-by-change cross product.
  **Expanded composition matrix (2026-08-09, D-081):** four training semantic
  graphs and four disjoint held-out role recombinations replace the original
  one-versus-one comparison. The factorized parser gets 4/4 held-out graphs;
  the trained whole-graph retriever fits 4/4 training graphs but gets 0/4
  held-out compositions. Familiar objects appear on both sides; only their
  goal/orientation/protection roles are recombined. **Broadened to the full
  combinatorial sweep (2026-08-14, D-117):** D-081's matrix was systematic
  in construction but a hand-picked sample (4 v 4); `full_role_matrix_cases()`
  instead enumerates every possible goal-pair over the object pool (96
  train, 84 held-out), with a checked guarantee that no held-out goal-pair
  ever appeared during training. Since the parser is deterministic
  rule-based code, the point of the larger sweep isn't statistical
  confidence — it's exercising the object-resolution logic against many
  more distinct strings than 4 examples could, looking for an
  unanticipated matching edge case. Result unchanged from D-081: factorized
  100%/100% (96/96, 84/84), both monolithic baselines 100%/0%. A genuine
  null result — no new edge case found — now backed by the full
  combinatorial space a 6-object pool allows rather than a small sample of
  it.
- **H5 — calibration:** calibrated uncertainty and abstention outperform forced
  binary feasibility decisions when evidence is ambiguous. **Sharpened
  2026-08-09 (D-078) — this needs a condition the original phrasing didn't
  name:** *only when the cost structure of being wrong is asymmetric in
  abstention's favor.* Measured directly, not assumed (see the full
  chronology below): outperforms when the true answer is SKIP and a wrong
  forced ATTEMPT is the expensive mistake (D-076/D-077); loses when the
  true answer is ATTEMPT and a wrong forced SKIP costs nothing in this
  project's reward shape (D-078) — abstaining still pays its fixed cost
  either way, so it only comes out ahead when what it's protecting against
  is itself costly.
  **First operational selective-prediction primitive (2026-08-08, D-073):**
  calibration now retains success/trial counts, derives a Wilson uncertainty
  interval, and makes a three-way attempt/skip/abstain decision. It attempts
  only when the interval's pessimistic endpoint has positive expected value,
  skips only when its optimistic endpoint is negative, and otherwise pays an
  explicit wait cost to abstain. Selective risk and coverage are reported
  separately. **The forced-versus-selective ablation itself built and run
  (2026-08-08, D-074/D-075):** calibrated on 20 real episodes, evaluated
  against reward-optimal labels derived from 80 disjoint held-out episodes
  (`bowl_destroyed`, wide onset timing). First real, observed result:
  `forced_risk=0.0, selective_risk=0.0, selective_coverage=0.75` — both
  methods answered every stratum correctly, but selective abstained on 1 of 4
  strata purely from limited calibration evidence, buying zero risk reduction
  at a real, measured 25% coverage cost. Honest negative/neutral evidence
  against an unqualified reading of H5 in this specific regime — D-071's
  strong per-intervention separation already made the point estimate correct,
  so there was nothing genuinely ambiguous for abstention to protect against
  here. **Given the fair test 2026-08-08 (D-076):** found a stratum whose
  true expected value sits close to the reward decision boundary
  (`bowl_destroyed`, `onset_step_bounds=(10, 100)`, true EV ≈ -0.41 from a
  200-episode held-out estimate — found by sweeping onset ranges and
  measuring directly, not guessed). Across 10 independent 20-episode
  calibrations against that fixed ground truth: the forced point-estimate
  baseline was wrong on 5/10 — a coin flip — while selective abstention was
  never confidently wrong (0/10), abstaining on 8/10 and answering correctly
  on the other 2. The first real, positive evidence in this project for H5's
  comparative claim, paired honestly with D-075's negative case: abstention
  doesn't help when the point estimate was already reliably correct, and
  does help, substantially, when it genuinely isn't — the trade-off (here,
  80% of coverage given up for a zero-wrong guarantee) still needs a
  downstream cost model to say whether it's worth it in a real deployment.
  **Built that model 2026-08-08 (D-077):** rather than inventing a new cost
  function, extended the same reward shape used everywhere else in this
  project (`+1.0` achieved, `-0.1 * steps_used` otherwise) to the ABSTAIN
  action as a small explicit wait cost, then re-ran D-076's exact
  stratum/seeds through it. Result: mean forced reward = -0.2044, mean
  selective reward = -0.0800 — selective wins clearly in the project's own
  reward units, not just on a risk/coverage count. Narrow, disclosed scope:
  this stratum's true value is negative-EV under either strategy, so this
  shows selective *loses less* here, not that it wins outright when the
  true answer is actually worth attempting — that sharper case (a stratum
  on the positive side of the boundary) is still untested.
  **Tested that sharper case 2026-08-09 (D-078), and forced won:** found a
  stratum whose true answer is ATTEMPT
  (`bowl_destroyed`, `onset_step_bounds=(10, 120)`, true survival ~0.73,
  true EV ~+0.07 from a 200-episode held-out estimate). Real result: forced
  was wrong on 3/10 seeds, but every wrong decision was a SKIP that costs
  exactly `0.0` in this reward shape — forced's mean reward is +0.0506,
  positive. Selective abstained on 8/10 (each costing `-0.1`) and was
  itself wrong once (a narrow interval landed entirely on the wrong side of
  the true boundary by chance) — selective's mean reward is -0.0728,
  negative. Forced clearly wins here, the opposite of D-077. The reason is
  a real asymmetry in this project's reward shape: a wrong ATTEMPT costs
  real reward, a wrong SKIP costs nothing (inaction is never penalized
  directly, only a failed action is) — so abstention's fixed cost is worth
  paying only when it protects against the expensive mistake. This
  completes H5's first honest, three-part picture (D-076 positive, D-077
  quantified, D-078 negative) and is why the hypothesis statement above now
  carries its condition explicitly instead of claiming an unconditional win.
  **Confirmed with bootstrap CIs, not 10-sample means (2026-08-21, D-120):**
  D-077/D-078 each drew their conclusion from 10 calibration seeds' raw
  mean on one stratum apiece. Reran both boundary strata with 30
  calibration seeds each and a paired bootstrap CI on the per-seed
  `(forced - selective)` reward difference (same protocol D-108 used for
  H3). Both signs held with real margin: negative-EV stratum, `(forced -
  selective)` CI `[-0.1995, -0.0632]` (selective wins); positive-EV
  stratum, CI `[0.0989, 0.1434]` (forced wins). The sign flip between
  strata is a robust property of the reward asymmetry, not noise from
  either individual 10-seed run.

## Success criteria

The project succeeds if it delivers a reproducible benchmark and demonstrates,
across multiple seeds, that the full agent improves feasible-goal completion
over a static-policy baseline while keeping hard-constraint violations below a
predeclared threshold. Feasibility accuracy alone is insufficient: estimates
must lead to better decisions. Oracle-feasibility performance defines the headroom.

**Run for the first time, 2026-08-10 (D-088):** `src/atr/evaluation/full_agent_benchmark.py`
now exists and works — a real, paired, multi-seed, bootstrap-CI comparison
of `static`, `oracle_feasibility` (the headroom reference this section
itself names), and the real `full_agent` pipeline (real language parsing,
real CLIP-perceived feasibility, a trained Q-table, real arm motion). The
first run's result was dominated by a newly-found CLIP perception gap (see
H1's update above), not a clean measurement of the policy's value.
**Fixed, and the real result obtained, 2026-08-10 (D-089):** with the
crop recalibrated, `full_agent` now matches `oracle_feasibility` exactly
on both `goals_achieved` and `wasted_steps`, every seed, and both beat
`static` on `wasted_steps` while matching it on `goals_achieved` — this
criterion's actual comparative claim, demonstrated with the real
perceptual pipeline for the first time. Scope still narrow and disclosed:
one env variant (ReplicaCAD-Humanoid), one scene (`kitchen_cabinet`), one
intervention (`chef_can_destroyed`), 15 seeds — not yet the general,
predeclared-threshold claim the roadmap's later phases ask for, but the
first real, positive instance of it rather than untested machinery.
**Broadened to a second intervention, 2026-08-10 (D-090):** checked
`temporary_obstacle` (reversible, feasibility-neutral distractor) with a
Q-table never trained on it — `static`/`oracle_feasibility`/`full_agent`
all achieve `goals_achieved=2.0`, `wasted_steps=0.0`, zero variance across
10 seeds, docs/10's "unnecessary adaptation" control passed cleanly with
real perception. Two disclosed gaps investigated and found non-actionable
rather than left untested by default: `kitchen_sink`'s reach/tray were
never calibrated for real arm motion (D-027), and `potted_meat_can` is
never checked post-attempt in the current fixed instruction's goal order.
**Extended to the second embodiment, privileged-state only, 2026-08-13
(D-108):** `TidyUp-ReplicaCAD-v1` (real ReplicaCAD apartment + Fetch mobile
robot) had never been checked this way — every prior confirmation of its
D-091–D-107 navigation-safety stack was one hand-placed scenario, not a
seed-varying benchmark. `static` vs `oracle_feasibility`, 30 paired seeds,
`bowl_destroyed`: identical `goals_achieved` seed-for-seed, and a real
(paired-bootstrap-significant) reduction in `wasted_steps` for
`oracle_feasibility`. Deliberately scoped below `full_agent`: this env's
camera is mobile, not fixed, so it has no CLIP calibration at all — running
`full_agent` here is a distinct, larger piece of work, not a natural
extension of this check.

## Threats to validity

Ground-truth simulator labels may make feasibility artificially easy; visual
changes may be detectable through shortcuts; templates may not reflect real
language ambiguity; and benchmark reward may encode the desired answer. Tests
must therefore include visual counterfactuals, paraphrases, held-out
interventions, and checks for representation leakage.
