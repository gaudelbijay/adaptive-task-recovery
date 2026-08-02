# Decisions

Lightweight architecture decision log. Stable research design is in `docs/`.

## D-034: Measured CLIP-vs-DINOv2 comparison recorded — evidence for I-004, deliberately not a selection

- **Date:** 2026-08-02
- **Status:** Accepted (as evidence; no model selected by this entry)
- **Decision:** Built the measured comparison
  `ai-notes/model-comparison-clip-vs-dinov2.md` against the criteria
  `docs/08-training-pipeline.md` already specifies for model selection
  (downstream utility, calibration, generalization, latency, memory,
  licensing, integration cost) — none of which had been recorded
  anywhere before this, despite D-020/D-023/D-027/D-029 already
  producing real accuracy/generalization/downstream-utility evidence for
  each model individually. New measurements taken directly, not assumed:
  latency and memory (isolated per-model subprocess, clean peak-RSS
  readings, 20 warmed-up calls each — CLIP ViT-B-32: 151.3M params, ~33ms/
  call, ~1287MB peak-RSS delta; DINOv2 ViT-S/14: 22.1M params, ~15ms/
  call, ~178MB delta); licensing (verified against each project's actual
  LICENSE file rather than assumed from memory — both MIT/Apache-2.0,
  permissive, not a differentiator, notably including catching that
  DINOv2's *original* 2023 release used a more restrictive license before
  Meta relicensed it, which would have been an easy, wrong assumption to
  carry forward); and one direct calibration run (DINOv2's probe via
  `predict_proba`, LOO, 12 examples: 100% accuracy, Brier 0.0001 — CLIP
  has no probability output to measure calibration against at all with
  its current interface, a real finding, not a gap papered over).
- **Reason:** I-004 (`ai-notes/issues_and_risks.md`) has been open since
  the project's reframing with no measured comparison behind it — real
  accuracy evidence existed per-model, but not against each other on the
  criteria the project's own training-pipeline doc says a selection needs.
  Building that now, while explicitly declining to select, follows the
  same pattern D-026 used for `Goal.condition`: produce real evidence,
  disclose it fully, but don't let building evidence quietly become
  making the decision the evidence is supposed to inform.
- **Consequences:** I-004 still open (not resolved by this entry) —
  `ai-notes/issues_and_risks.md`'s mitigation note ("choose only after
  task schema and compute budget are known") stands: D-013's review still
  hasn't resolved. Whoever makes that call later has the numbers now.
  Notable gap surfaced, not just measured: DINOv2 has never been tested
  against the `kitchen_sink` scene variant (D-027) despite the code
  supporting it — CLIP has 2-scene validation, DINOv2 has 1.

## D-033: ManiSkill3 formally selected as the primary simulator, closing I-003 without an Isaac Lab spike

- **Date:** 2026-08-02
- **Status:** Accepted
- **Decision:** Formally selects ManiSkill3 as the project's primary
  humanoid-capable simulator. D-006 required a simulator spike before
  committing to simulator-specific architecture — it did not require
  evaluating a second candidate, and I-003 (`ai-notes/issues_and_risks.md`)
  had already flagged that the case for treating the ManiSkill3 evidence
  as sufficient was "considerably stronger... than at last write-up."
  That evidence, as of this entry: humanoid support, deterministic
  seeding, privileged state, object-level interventions, RGB-D
  observations, and reach/grasp all confirmed (D-009–D-011); five
  further build-up stages built and stress-tested on it, across four
  robot/scene combinations (D-013–D-029); one real upstream bug found,
  root-caused against a known GitHub issue, and worked around rather
  than guessed at (D-022); one real kinematic limit confirmed with a
  proper verified IK solver rather than assumed (D-024/D-028); one real
  platform gap found and worked around (`mplib` doesn't build on Apple
  Silicon, `pinocchio` does) — with nothing disqualifying turning up
  across any of it.
- **Reason:** An Isaac Lab spike would be a second full simulator
  integration — new install, new asset validation, new platform-gap
  discovery process, on a Low-severity open question that D-006 never
  actually required resolving via head-to-head comparison. Weighed
  against seven weeks of accumulated, working, tested ManiSkill3-specific
  evidence, spiking a second simulator now would cost real time for a
  comparison this project doesn't need to make to keep moving —
  the question D-006 asked ("does a viable simulator exist") has been
  answered affirmatively and repeatedly, not left open.
- **Consequences:** I-003 closed (moved to "Resolved or superseded" in
  `ai-notes/issues_and_risks.md`). ManiSkill3/`sapien`-specific code can
  now be treated as a real, if still spike-stage, architectural
  commitment rather than a placeholder pending a simulator decision —
  though it still can't move into `src/atr/` until D-013's separate,
  still-open schema review resolves (see D-032). Isaac Lab remains a
  live option later if a specific ManiSkill3 limitation actually blocks
  something (e.g. D-022's rendering bug, if it turns out to matter more
  than currently worked around) — this decision closes the open
  *question*, not the door.

## D-032: `src/atr/`, `configs/`, `data/` scaffolded — structure only, no code migrated

- **Date:** 2026-08-02
- **Status:** Accepted
- **Decision:** Created the directory structure the `STATUS.md` todo has
  named since the project's reframing: `src/atr/` (with an `__init__.py`
  and README explaining why it's empty), `configs/`, and `data/`
  (`scripts/` and `tests/` already existed). Deliberately did **not**
  move any code out of `spikes/task_schema_draft/` — not `goal_graph.py`,
  not `oracle_feasibility.py`, nothing. `data/` added to `.gitignore`
  (all but its own README) since datasets don't belong in git history.
  `pyproject.toml` left untouched — no packaging config added for an
  empty package; that's a decision to make once there's real code to
  package, not before.
- **Reason:** `ai-notes/review-request-task-schema.md` (sent this same
  day, D-030's follow-up) explicitly asks the teammate whether D-013's
  schema is "ready to move from `spikes/task_schema_draft/` into
  `src/atr/` as committed architecture, or needs changes first." Moving
  the code into `src/atr/` before that review lands would answer the
  review's own central question by fait accompli, undermining the point
  of having sent it. Confirmed directly with the user before proceeding
  rather than assuming scope, since this was a genuine fork with a real
  consequence either way, not a judgment call between two reasonable
  interior details.
- **Consequences:** `src/atr/` is import-empty; nothing in this project
  currently runs from it. Once the D-013 review resolves (accepted
  as-is, accepted with changes, or sent back), the reviewed pieces move
  here for real, at which point `pyproject.toml` needs an actual `atr`
  package entry and the interface-versioning todo (goal graphs,
  feasibility beliefs, skills, logs) becomes concrete rather than
  hypothetical.

## D-031: Dependabot vulnerability triage — all 9 flagged packages fixed, one required a `sapien` bump first

- **Date:** 2026-08-02
- **Status:** Accepted
- **Decision:** GitHub flagged 38 Dependabot alerts (28 high) after the
  D-030 push. Ran `pip-audit` against `requirements-maniskill.lock.txt`
  locally (no `gh`/GitHub-API auth available in this environment) and
  confirmed 50 known-CVE entries across 9 packages: `click`, `GitPython`,
  `idna`, `lxml`, `pillow`, `Pygments`, `requests`, `setuptools`, `urllib3`
  — all transitive dependencies, none of them `mani_skill`/`sapien`/`torch`
  or other packages this project's own code imports directly. Bumped 8 of
  the 9 immediately (`click` 8.3.1→8.4.2, `GitPython` 3.1.46→3.1.57, `idna`
  3.11→3.18, `lxml` 6.0.2→6.1.1, `pillow` 12.1.1→12.3.0, `Pygments`
  2.19.2→2.20.0, `requests` 2.32.5→2.34.2, `urllib3` 2.6.3→2.7.0). The
  9th, `setuptools` (81.0.0→83.0.0 fixes PYSEC-2026-3447), broke test
  collection outright at first attempt: confirmed directly, by downloading
  and inspecting the wheel rather than assuming, that setuptools removed
  `pkg_resources` entirely as of 82.0.0, and `sapien` 3.0.2 — a core,
  load-bearing dependency of every environment in this project — imports
  `pkg_resources` at module load time (`ModuleNotFoundError` before a
  single test could run). Checked for a real fix rather than settling for
  the tradeoff: `sapien` 3.0.3 (one patch release ahead, released after
  3.0.2) drops the `pkg_resources` import entirely (confirmed by
  inspecting its wheel too — zero references, vs. 3 in 3.0.2), pulling in
  `importlib_resources` instead. Bumped both `sapien` and `setuptools`
  together; full suite re-run clean (97 passed, 417s). All 9 packages now
  fixed, lock file regenerated.
- **Reason:** Same standard as D-022 elsewhere in this project — check
  whether an apparent dead end is actually one (a newer patch release existed
  the whole time) before settling for a disclosed-but-unfixed gap. The
  first pass here nearly shipped `setuptools` held back at 81.0.0 as a
  "genuine tradeoff"; it wasn't one, it was an incomplete search — the
  actual fix was a one-patch-version `sapien` bump, no different in kind
  from D-030's own dependency work.
- **Consequences:** Zero open Dependabot-flagged packages in the lock
  file as of this entry. `importlib_resources==7.1.0` added as a new
  transitive dependency (pulled in by `sapien` 3.0.3). GitHub's Dependabot
  UI wasn't directly queryable in this environment (`gh` CLI not
  installed, no `GITHUB_TOKEN`) — this triage was done by auditing the
  lock file locally with `pip-audit` instead; the original 38/28-high
  count is GitHub's own report, not independently re-verified against
  Dependabot's exact dedup/scoring logic, though the 9-package/50-CVE
  `pip-audit` result is consistent with it in substance.

## D-030: Professional file/function naming pass, and de-duplicating `train_q_table`

- **Date:** 2026-08-02
- **Status:** Accepted
- **Decision:** Renamed several spike files and one duplicated function pair
  to names that describe what they contain rather than an arbitrary stage
  label, using `git mv` throughout so history is preserved:
  `language.py` → `instruction_parser.py`, `vision.py` →
  `clip_feasibility.py`, `representation.py` → `dinov2_probe.py`, and
  `_capture_episode_subprocess.py` → `capture_episode_subprocess.py`
  (dropped the leading underscore — it's invoked directly as a subprocess
  entry point by `dinov2_probe.py`, not a private helper). Matching test
  files renamed the same way (`test_vision.py` →
  `test_clip_feasibility.py`, etc.). Also collapsed a real duplication
  D-029 introduced: `rl_policy.py`'s Q-learning training loop had been
  copy-pasted into `end_to_end.py` as
  `train_q_table_for_replicacad_humanoid()`, differing only in which
  env/goals/attempt-function got passed in — and D-029's own
  `_wait()` timing fix had to be rediscovered and reapplied there
  independently (see D-029), which is exactly the failure mode duplicated
  logic invites. Replaced both with one parameterized `train_q_table()` in
  `rl_policy.py` taking `make_env`/`graph`/`tray_slots`/`attempt_goal_fn`/
  `intervention_kinds`/`onset_step_bounds`, plus thin per-env wrappers
  (`train_q_table_canonical()`, `train_q_table_replicacad_humanoid()`)
  that just supply those arguments. `train_q_policy()` renamed to
  `train_q_table_canonical()` to match.
- **Reason:** User feedback: file and function names should describe their
  content professionally, not read as an ordered list of build stages
  (`vision.py`, `language.py` say nothing about *what's inside* — CLIP
  zero-shot classification vs. regex instruction parsing — and stage
  numbering belongs in `docs/00-project-overview.md`'s build-up order, not
  the filesystem). Fixing the naming surfaced the `train_q_table`
  duplication along the way; worth fixing at the same time rather than
  renaming both copies and leaving the drift risk in place.
- **Consequences:** All imports, test imports, and cross-references in
  `ai-notes/` and `docs/` updated to match (verified by repo-wide grep for
  every old name). No behavior change — same algorithm, same test
  coverage, full suite re-run green after the rename. Found along the way:
  BSD `sed` (macOS default) silently matches zero occurrences on `\b`
  word-boundary patterns rather than erroring — GNU-only syntax — so the
  `.md`-file bulk-replace pass had to be redone without `\b` after an
  initial silent no-op.

## D-029: Stage 6 — everything combined into one real episode, nothing privileged in the live decision loop

- **Date:** 2026-08-02
- **Status:** Accepted (toy-scale, one episode type — same caveats as
  every stage this builds on)
- **Decision:** Built `end_to_end.py`, completing the build-up order in
  `docs/00-project-overview.md`. For each goal in a real episode:
  `parse_instruction()` (D-019/D-026) turns the instruction into a
  `GoalGraph`; a real rendered frame plus `visual_object_exists()` (D-020)
  judges feasibility — not a privileged-state read; a Q-table trained by
  `train_q_table_replicacad_humanoid()` (same algorithm as D-025,
  retrained for this env's parser-generated goal ids) decides attempt vs.
  skip from that *perceived* feasibility; `attempt_goal()` executes the
  decision with real arm motion, unchanged. Result: `potted_meat_can`
  (perceived feasible, matches oracle) gets attempted and achieved;
  `master_chef_can` (perceived infeasible after the scripted destruction,
  matches oracle) gets skipped at zero cost — the same H2 result every
  earlier stage produced, now with nothing privileged in the live decision
  path. Training itself still reads privileged state — a deliberate,
  disclosed choice: training the decision *rule* doesn't need real pixels
  for this toy case, and training against real rendered rollouts would
  need hundreds of render-producing resets, which D-022's confirmed
  upstream bug makes impractical. Found and fixed the same real bug
  D-025 already found once, hit again by not applying its own fix here:
  skipping the first goal via exploration shortens elapsed time before the
  second goal's feasibility check, producing a stale read relative to when
  the intervention actually fires — confirmed directly (a negative
  Q-value for a feasible goal, which should never happen), fixed the same
  way (`_wait()`, keeping elapsed time consistent regardless of action).
- **Reason:** The last stage named in the build-up order. Worth doing as
  an actual integration, not just five demonstrations that happen to share
  a codebase — the interesting failure mode (stale timing assumptions
  breaking when exploration enters the picture) only showed up once
  real pieces were actually wired together and exercised end-to-end.
- **Consequences:** Toy-scale in every way its component stages already
  were: one instruction, one scene layout (`kitchen_cabinet`, the only one
  clip_feasibility.py's calibration and attempt_goal's reach configs both cover),
  two goals, privileged-state training. `dinov2_probe.py`'s DINOv2 probe
  (D-023) was deliberately not wired into this same live loop — it needs a
  pre-fit probe from multiple examples, not a single-frame judgment like
  CLIP, and wiring it in would add complexity disproportionate to what
  this stage needed to show; it remains a separately-validated alternative
  perceptual backend. This closes the build-up order from
  `docs/00-project-overview.md` — everything from here is either genuine
  scaling work or the still-open teammate review
  (`ai-notes/review-request-task-schema.md`).

## D-028: D-024 retried with a proper analytic-Jacobian IK solver — confirmed unreachable, not a solver artifact

- **Date:** 2026-08-01
- **Status:** Accepted — D-024's grasp-confirmation gap remains, but now
  backed by a much stronger negative result, plus a real, reusable,
  validated IK tool (`ik_solver.py`) for future use
- **Decision:** D-024's finite-difference IK was unreliable (11cm one run,
  57cm another, identical inputs). Rebuilt it properly on `pinocchio`
  against G1's actual URDF (`ik_solver.py`): a real analytic Jacobian via
  `pin.computeFrameJacobian`, damped least-squares (not plain
  pseudo-inverse, more stable near singularities). Verified before
  trusting it: pinocchio's local-frame forward kinematics for
  `right_tcp_link` matches `agent.right_tcp.pose.sp.p - agent.robot.pose.sp.p`
  to 5 decimal places (G1's base has zero rotation when placed via
  `sapien.Pose(p=...)`, confirmed not assumed). Result: **fully
  deterministic** (identical distance across 5 repeated runs, unlike the
  finite-difference version) and, searched with random-restart
  initialization across a wide, floor-clearance-checked set of candidate
  base positions (32 candidates at 4 radii × 8 angles around each object,
  plus the original position), **cannot bring the tcp within ~13cm of
  either target object** in the "kitchen_cabinet" scene. Not joint-limit
  bound (checked directly — no arm joint sits at its limit at convergence).
  Real, physical, contact-force-verified grasp needs roughly <5cm (D-024's
  own finding: zero contact force registered even at ~10-11cm). Also
  found: the two objects are ~0.6m apart, wider than the arm's functional
  reach envelope from any single standing position — no repositioning can
  bring *both* within range simultaneously, and closer positions than the
  original (raycast-clearance-checked, tried directly) made the residual
  distance *worse*, not better, since they force awkward elbow/shoulder
  angles.
- **Reason:** Direct follow-up to D-024 per explicit request, using a
  principled tool (real analytic IK) instead of retrying the same
  unreliable technique. Distinguishing "the solver was bad" from "the
  target is genuinely out of reach" required building the better solver
  first — couldn't have concluded this with confidence from D-024's
  evidence alone.
- **Consequences:** Real contact/tactile grasp confirmation remains
  unimplemented for these specific objects from this specific base
  position — teleport-on-success is unchanged, same as D-024 concluded.
  What's different now: this is a confirmed structural limit (arm length
  vs. object separation, checked from every reasonable standing position),
  not an open question that a better solver might still resolve.
  `ik_solver.py` is kept as a real, tested, reusable module (D-028's own
  tests verify it against ManiSkill's kinematics and lock in this
  unreachability finding as a regression test) — useful if this project
  ever needs real IK for a *different* object/scene combination where the
  geometry might actually allow it.

## D-027: A second calibrated scene layout for clip_feasibility.py/dinov2_probe.py — not a single-scene-only demonstration anymore

- **Date:** 2026-08-01
- **Status:** Accepted (still toy-scale — two scenes, not a distribution)
- **Decision:** Added "kitchen_sink", a second calibrated apartment layout
  (`build_config_idx=55`, found searching under the *real* two-pin
  `torch.manual_seed` pattern D-021 established — a naive single-pin search
  gives different, wrong results, the same lesson D-021 already learned
  applied here again) to `tidy_up_env_replicacad_humanoid.py`'s new
  `_SCENE_CONFIGS` dict, selected via a `scene_variant` constructor
  argument (default `"kitchen_cabinet"`, so every existing call site and
  test is unaffected). Camera and crop calibration used a more precise
  method this time: projected each object's known world position through
  the render camera's own intrinsic/extrinsic matrices to get exact pixel
  coordinates, rather than finding crops by visual inspection alone (the
  original "kitchen_cabinet" method) — needed because `potted_meat_can`
  turned out to be sitting inside a sink basin in this layout, small and
  easy to miss by eye. `clip_feasibility.py`'s `_OBJECT_VISUAL_CONFIG` is now keyed
  per scene variant; `visual_object_exists()` and
  `dinov2_probe.py`'s `collect_labeled_examples()` both take an optional
  `scene_variant` argument, defaulting to `"kitchen_cabinet"`. Verified:
  zero-shot CLIP matches oracle feasibility on "kitchen_sink" the same way
  it did on the original scene (`test_vision_kitchen_sink.py`).
  Deliberately *not* recalibrated for this layout: reach configs, tray
  position, or the goal graph — "kitchen_sink" is vision/rendering-only;
  using it with the reach-dependent policy baselines is untested and out
  of scope.
- **Reason:** Direct answer to the review document's caveat that
  clip_feasibility.py/dinov2_probe.py were validated on a single scene layout only
  — not a full generalization test, but a genuine second data point instead
  of zero.
- **Consequences:** `test_vision_kitchen_sink.py` uses subprocess-isolated
  capture (like dinov2_probe.py), not in-process rendering like
  test_clip_feasibility.py — test_clip_feasibility.py already spends this process's entire
  D-022 render-producing-reset budget (2) on "kitchen_cabinet"; testing a
  second variant in the same process would exceed it. Still two scenes, not
  a real distribution over layouts — the "not a generalization test" caveat
  is weaker now, not gone.

## D-026: Ordering/priority and conditional goals — language grammar and a proposed schema extension

- **Date:** 2026-08-01
- **Status:** Ordering/priority: Accepted (uses existing schema fields, no
  new decision needed). Conditional goals: Proposed, same "needs review"
  status as D-013 itself — `Goal.condition` is a new schema field, not
  something to accept unilaterally right before asking for exactly that
  review.
- **Decision:** `instruction_parser.py` now parses "first put the mug on the tray,
  then put the bowl on the tray" into sequential `Goal.priority` values (0,
  1, ... in order of appearance among order-marked goal clauses; unmarked
  clauses keep priority=0, so every existing instruction_text still parses
  identically). Also added a conditional-goal pattern: "if the blue bowl is
  destroyed, put the backup bowl on the tray instead" sets a new,
  PROPOSED `Goal.condition: tuple[str, bool] | None` field — (object_id,
  required_exists) — checked in `goal_feasible()` before the goal's own
  target object even matters. Real design problem found and solved: the
  generic clause splitter breaks any comma immediately before a recognized
  verb ("put"), which is exactly the shape of "if X is Y, put Z on the
  tray" — extracting conditional clauses in a separate pass, before the
  generic splitter runs on what's left, avoids the conflict entirely
  (see instruction_parser.py's module docstring for the full explanation).
- **Reason:** Direct request to fix the "ordering/priority and conditional
  goals are unimplemented" caveat from `ai-notes/review-request-task-schema.md`.
  Ordering was safe to just build (existing fields, no new schema
  surface). Conditional goals needed a real judgment call: build it
  properly and test it, but don't quietly promote it to "accepted" schema
  status when the entire point of the review request is to gate exactly
  this kind of decision — so it's built, tested, and honest about still
  needing that review, not either skipped or smuggled in as settled.
- **Consequences:** `docs/04`'s "preferences" (soft, non-binding wishes)
  remain entirely unimplemented — no schema field exists for them, adding
  one is a similarly-sized schema decision, not attempted here without a
  driving case. `ai-notes/review-request-task-schema.md` updated to flag
  `Goal.condition` as a second thing needing your teammate's review, not
  just D-013's original fields.

## D-025: First learned policy — tabular Q-learning discovers the feasibility rule from reward, not from being told it

- **Date:** 2026-08-01
- **Status:** Accepted (toy-scale — 2 goals, 3 meaningful states, same
  caveats as every other toy-scale demonstration in this project)
- **Decision:** Built `rl_policy.py`: tabular Q-learning over
  `(goal_id, feasible) -> {SKIP, ATTEMPT}`, trained across 120 randomized
  episodes (intervention present or not, timing varied) using real
  environment rollouts — real arm motion via `attempt_goal()` from
  `policy_baselines.py`, unchanged. Trains in ~19s on CPU. Result: the
  learned greedy policy converges to exactly "attempt iff feasible" —
  `feasibility_aware_policy`'s hard-coded rule — without ever being told
  that rule, and matches it exactly head-to-head (same goals achieved, zero
  wasted steps vs. static's 25). A real bug found and fixed while building
  this: epsilon-greedy exploration can choose to skip the first goal, which
  (unlike the deterministic baselines, which always attempt it) shortens
  elapsed time before the second goal's feasibility check — occasionally
  reading "feasible" correctly at check-time, then having the intervention
  fire mid-attempt, producing a systematic negative bias in one Q-value
  (confirmed: `("place_bowl", True)` converged to -0.98 instead of +1.0 at
  n_episodes=120). Fixed by making SKIP consume the same elapsed time an
  attempt would have (`_wait()`), keeping the state observation
  non-stale regardless of which action gets explored.
- **Reason:** Stage 5 of the build-up order in
  `docs/00-project-overview.md` — replace the scripted/oracle policies with
  one that's actually learned. Deliberately scoped to the *decision* layer
  (attempt vs. skip a goal) matching this project's research question
  throughout, not a learned motor policy — low-level control (the reach
  phase) is untouched. Operates entirely on privileged state, no
  rendering, so D-022's confirmed upstream rendering bug doesn't apply
  here at all.
- **Consequences:** This is 3 Q-table entries, not a general RL result —
  the state space here is trivial by construction (2 goals × exists/not).
  What it does demonstrate cleanly: the same behavior D-014 got by
  hard-coding a rule can instead be recovered by trial-and-reward learning,
  on real environment rollouts, in under 20 seconds on CPU with no GPU
  needed.
  Extending this to a real state space (vision/representation-derived
  feasibility estimates instead of privileged-state ones, more goals,
  ordering/priority) is future work, not attempted here.

## D-024: Real contact/tactile grasp confirmation attempted, found genuinely infeasible with current tooling, not implemented

- **Date:** 2026-08-01
- **Status:** Investigated, not implemented — a documented limitation, not
  a silent gap
- **Decision:** Requested addition: alongside vision, confirm grasp success
  via real contact forces (G1's built-in `right_hand_is_grasping()`) during
  the reach phase, keeping teleport-on-success for final placement only.
  Found this isn't achievable with current tooling: G1's existing reach
  configs (used everywhere in this project) only ever bring the arm to
  ~45cm from the target object — fine for teleport-on-success, which never
  needed real precision, but nowhere near contact range. Built a
  closed-loop numerical-Jacobian IK solver (finite-difference Jacobian,
  damped least-squares step) to close that gap adaptively, since G1 has no
  Cartesian controller or analytic IK exposed in ManiSkill (D-016). It
  converged inconsistently: the *same* starting base position and joint
  config converged to 11cm from the object in one run and 57cm in another,
  no code difference between runs. At the distances it did reliably reach,
  closing the fingers produced zero contact force — genuinely no touch, not
  a threshold issue. Tried moving G1's base closer to the object (raycast
  floor-clearance-checked, same method as D-018) — didn't resolve the
  underlying convergence instability. Stopped here rather than continuing
  to iterate on an unreliable numerical method or building a proper IK
  pipeline (e.g. wiring `pinocchio` — already a dependency, used for
  Panda's Cartesian controller in the original spike — into a real
  analytic-Jacobian solver for G1 specifically) without a clear signal
  that's worth the effort for this project's actual research question.
- **Reason:** Genuinely attempted, not deprioritized on a guess — the user
  asked directly, and a real effort (grid search, closed-loop IK, base
  repositioning) was made before concluding this is a bigger problem than
  "recalibrate a constant."
- **Consequences:** teleport-on-success remains the manipulation
  abstraction throughout this project, unchanged — grasp mechanics were
  never load-bearing for any existing result (H2/H3, clip_feasibility.py,
  dinov2_probe.py all operate on privileged/perceptual existence, not
  grasp success). If real contact-based confirmation is needed later, the
  actual path is a proper analytic-Jacobian IK solver built on `pinocchio`
  (already installed) against G1's real URDF kinematic chain — not another
  attempt at finite-difference numerical IK, which is what proved
  unreliable here.
  **Follow-up (D-028, 2026-08-01):** built exactly that proper solver and
  retried. Confirms this is a genuine kinematic limit, not a solver
  problem — see D-028.

## D-023: First self-supervised representation layer — DINOv2 linear probe, worked around a confirmed dependency bug rather than blocking on it

- **Date:** 2026-08-01
- **Status:** Accepted (toy-scale, single-scene — same caveats as D-020's
  vision layer, see Consequences)
- **Decision:** Built `dinov2_probe.py`: `dinov2_embed()` extracts a
  384-dim CLS-token embedding from DINOv2 ViT-S/14
  (`facebookresearch/dinov2`, self-supervised, no text/labels in its
  training — genuinely different from D-020's CLIP, which is
  language-supervised). `fit_and_evaluate_probe()` fits a logistic-regression
  linear probe and evaluates it with leave-one-out cross-validation. Result
  on 8 examples (master_chef_can, 4 present / 4 absent): 100% LOO accuracy
  — the representation linearly separates object-presence at least as well
  as D-020's zero-shot CLIP did on the same task.
  D-022's confirmed upstream rendering bug (open, no fix) means more than
  ~2 render-producing resets in one process can't be trusted — a real
  obstacle to collecting enough labeled examples for a probe. Worked around
  it rather than either blocking on it or silently risking corrupted data:
  `capture_episode_subprocess.py` captures exactly one labeled example per
  subprocess invocation, so every capture is "the first" render-producing
  reset from the OS's point of view, staying inside the verified-safe zone
  every time. `collect_labeled_examples()` shells out to it per example.
- **Reason:** Stage 4 of the build-up order in
  `docs/00-project-overview.md` — swap in a representation learned from
  unlabeled data, once stage 3 (any working pretrained model) works.
  Deliberately checked whether a representation with *no* language
  supervision still supports this judgment, not just a bigger CLIP.
- **Consequences:** D-021 pinned this env's scene layout for good reason
  (G1's placement is only valid on one apartment layout), which means every
  example collected here is visually almost the same scene — this is not a
  test of representation *generalization* (different objects, layouts,
  lighting), only of whether DINOv2's embedding linearly separates
  presence/absence at all, on the one scene currently renderable. 100%
  accuracy on 8 examples of a genuinely easy, low-noise task should not be
  read as "DINOv2 solves feasibility perception" — it's the minimum bar
  this stage needed to clear before being worth building on. The
  subprocess-per-example pattern is slow (~6s/example) and would not scale
  to a real training set; if this stage ever needs more than toy-scale
  data, that means either fixing/upgrading past D-022 or finding a
  different data-collection strategy, not more subprocesses.

## D-022: Render-producing-reset desync — confirmed as a known, open, unfixed upstream ManiSkill3 bug

- **Date:** 2026-08-01
- **Status:** Accepted as a documented, guarded, confirmed-upstream issue —
  not fixable at this project's level, not a guess anymore
- **Decision:** Followed D-021's rendering finding to an actual root-cause
  attempt. Confirmed properties, each tested directly rather than assumed:
  unrelated to seed (identical `seed=0` config, repeated); reproduces with
  the *same* env instance across repeated `reset()` calls, not just fresh
  `gym.make()` instances; reproduces with `options={"reconfigure": True}`
  forced on every reset; unaffected by `sapien.render.clear_cache()`;
  `ambient_light` and light-entity count identical across instantiations
  (ruled out a lighting-value explanation); simple brightness/contrast
  normalization of the crop does not fix `clip_feasibility.py`'s resulting
  misclassification; reproduces on **both** `tidy_up_env_replicacad.py` and
  `tidy_up_env_replicacad_humanoid.py` (rules out anything specific to
  either env's own code). Visually confirmed the failure mode is not just
  "darker" — later renders sometimes show entirely different furniture
  geometry while privileged object positions stay correct, i.e. the
  rendered scene graph desyncs from the physics scene.
  **Then checked whether this is a known upstream bug rather than stopping
  at an educated guess:** it is.
  [haosulab/ManiSkill#1150](https://github.com/haosulab/ManiSkill/issues/1150)
  ("Observations turn green after reset in PickSingleYCB-v1 and
  PickClutterYCB-v1 environments on macOS") reports the same shape of bug —
  macOS-only, specifically the YCB-object-loading environments (not simple
  primitive ones like PickCube-v1), breaking after the 2nd or 3rd reset
  within one process. Filed October 2025 (per GitHub numbering/timing),
  still **open**, no maintainer fix or workaround, no branches or PRs
  addressing it. Both our envs load real YCB objects via ReplicaCAD, so
  this matches. Installed version here: `mani_skill==3.0.0b22`.
  Given a confirmed, still-open bug in the library itself with no known
  workaround from its own maintainers, patching it in this project isn't a
  realistic option — instead: both env files count render-producing resets
  (`_render_producing_reset_count`, module-level, per env class) and
  `warnings.warn()` past the second one in a process, so a silently-wrong
  render becomes a loud warning instead of a trusted one.
- **Reason:** After D-021's fix, this was the one remaining thread from the
  "fix all these things" / "fix what's still needed" asks. Worth
  distinguishing "I couldn't find the cause" from "this is a confirmed,
  open bug in a dependency, unfixed even by its own maintainers" — the
  second is a much stronger, more actionable thing to have on record than
  the first.
- **Consequences:** `clip_feasibility.py` results are only trustworthy for the first
  one or two render-producing resets of these envs in a process — verified
  by inspecting saved frames directly (`tests/drafts/test_clip_feasibility.py`'s two
  cases both checked this way, see that file's docstring), not merely
  assumed safe. A batch script or notebook that constructs many such env
  instances in a loop and renders each one will hit this and should not
  trust results past the warning without visually spot-checking frames.
  Genuinely not resolvable here; revisit by checking whether
  haosulab/ManiSkill#1150 has closed on a future ManiSkill3 upgrade.

## D-021: Fixed the scene-layout generalization gap D-020 found — and found a deeper, unresolved one

- **Date:** 2026-07-31
- **Status:** Accepted (the object-placement fix); the rendering finding
  below is explicitly *not* resolved — see Consequences
- **Decision:** Direct follow-up to D-020's finding #4. Root cause:
  `ReplicaCADRearrangeSceneBuilder` draws from torch's *global* RNG at two
  independent points — once for `sample_build_config_idxs()` (which
  apartment) and again inside `initialize()` for which YCB objects are
  actually placed versus hidden at z=-10000 — neither tied to this env's own
  `_episode_rng`. Confirmed both `tidy_up_env_replicacad_humanoid.py` and
  `tidy_up_env_replicacad.py` (same scene_builder_cls) were affected;
  `env.reset(seed=2)` on the Fetch variant hid *both* of that env's goal
  objects outright. Fixed in both files: force
  `build_config_idxs=[59]`/`init_config_idxs=[0]` (the config `reset(seed=0)`
  happened to sample before this fix existed) and call
  `torch.manual_seed(0)` immediately before both scene-construction calls
  (`_load_scene`, `_initialize_episode`), decoupling scene layout entirely
  from the `seed` argument. Verified with a new regression test in each
  env's test file (`test_scene_layout_reproducible_across_seeds`): all four
  target objects now land at byte-identical positions across seeds
  {0, 2, 7/15, 42}.
  **Separate finding, not resolved:** while verifying this fix against
  `clip_feasibility.py`, rendered frames sometimes came out visibly darker/differently
  exposed than the known-good look — but this turned out to be unrelated to
  `seed` at all. Creating the *same* env config (`seed=0`, every field
  identical) repeatedly in one Python process gave a correctly-lit render on
  the first instantiation and a measurably darker one (mean pixel value 114
  vs 39) on the second and third, even though the underlying object
  positions were confirmed identical. This looks like renderer/scene-graph
  state not being fully released between `env.close()` and the next
  `gym.make()` for this env+render config, not a scene-layout issue.
  **Follow-up (D-022, 2026-08-01):** investigated this properly rather than
  leaving it as a guess — root cause not found, but narrowed a lot and now
  guarded with a runtime warning. See D-022.
- **Reason:** D-020 explicitly flagged this as unfixed; fixing it removes a
  real correctness gap in both real-scene environments, not just the one
  under vision-layer development.
- **Consequences:** Object placement and reachability are now genuinely
  seed-independent in both ReplicaCAD envs — this closes D-018's correction
  note. The rendering/instantiation-order finding is new, real, and
  unresolved; do not assume `clip_feasibility.py`'s calibration holds if this env is
  instantiated with `render_mode` set many times in one process (e.g. a
  batch evaluation loop) without further investigation first.

## D-020: First vision layer — zero-shot CLIP, and two real bugs it surfaced

- **Date:** 2026-07-31
- **Status:** Accepted (single-scene proof of concept, not a general result —
  see Consequences)
- **Decision:** Built `clip_feasibility.py`: `visual_object_exists(frame, object_id)`
  judges object presence from a rendered camera frame using zero-shot CLIP
  (`open_clip`, ViT-B-32, OpenAI weights — no training), instead of reading
  `WorldState.exists` from the simulator. New dependency, installed clean on
  Apple Silicon (unlike `mplib`/`habitat-sim`); `requirements-maniskill.lock.txt`
  regenerated. Four things had to be found empirically before this worked at
  all, none of them assumed going in:
  1. Whole-frame CLIP similarity barely moves when an object is actually
     removed (measured delta ~0.01, sometimes the wrong sign, across 20
     seeds) — the object is too small a fraction of a cluttered frame. A
     tight crop around the object's known on-screen location (fixed camera,
     fixed crop — camera calibration, not a live 3D-position read) fixed this.
  2. `tidy_up_env.py`'s "objects" are plain colored boxes (`build_box`
     primitives), not the real objects they're named after — zero-shot CLIP
     correctly can't recognize "a blue bowl" in a picture of a blue cube,
     because there isn't one there. Switched calibration to
     `tidy_up_env_replicacad_humanoid.py` instead, which has real
     photorealistic YCB-scanned objects (D-017/D-018).
  3. **A real, previously-latent bug:** `_trigger_intervention()`'s
     `chef_can_destroyed` branch removed the object from physics but never
     called `self.scene.update_render()` — every existing consumer of this
     env reads privileged state, not pixels, so a stale render went
     unnoticed until this was the first code to actually look at a frame
     after a removal. Fixed by adding the same `update_render()` call the
     `temporary_obstacle` branch already had.
  4. **A second real, previously-latent bug, found but not fixed:** G1's
     hardcoded base pose and camera in `tidy_up_env_replicacad_humanoid.py`
     are calibrated for exactly one apartment layout.
     `ReplicaCADSetTableTrain` loads a different room per seed — rendering
     seed=2 placed G1 next to a couch and a bicycle, nowhere near the cans.
     Every prior test of that env (D-018) only ever used seed=0, so this was
     never caught until vision work rendered and looked at other seeds.
     `tests/drafts/test_clip_feasibility.py` is deliberately seed=0-only because of
     this. Generic prompts ("a photo of a green can") also measurably
     underperformed specific/iconic ones ("a photo of a Spam can") — not a
     bug, but a real, documented CLIP behavior worth knowing.
  Final result at seed=0: 4/4 correct (both objects, before and after the
  intervention) — matches oracle feasibility on every case tested.
- **Reason:** Stage 3 of the build-up order in
  `docs/00-project-overview.md` — "vision, simplest version first... any
  working pretrained visual model" — the actual point of which is comparing
  a real (imperfect) vision signal against the privileged-state oracle, per
  docs/01's "Oracle-feasibility performance defines the headroom."
- **Consequences:** This is 4 data points from one scene layout, not a
  statistically meaningful accuracy claim — do not cite this as "CLIP
  achieves X% feasibility accuracy" in any general sense. `_OBJECT_VISUAL_CONFIG`
  is hand-calibrated per object (crop + prompt) for this exact camera pose;
  it is not a general object detector and raises rather than guessing for
  any object without a calibrated entry. Finding #4 (seed-generalization gap
  in G1 placement) is a real correction to D-018's implicit scope — that
  work was only ever validated at seed=0, not stated clearly enough there.
  Fixing scene-layout generalization is a separate, later problem, not
  addressed here.

## D-019: First language layer — instructions parsed into goal graphs, not hand-written

- **Date:** 2026-07-30
- **Status:** Accepted (controlled grammar, not open-ended NLU — scoped
  intentionally, see Consequences)
- **Decision:** Built `instruction_parser.py`: `parse_instruction(text, known_objects)`
  turns an instruction sentence into a `GoalGraph` via a controlled grammar
  covering the two forms every existing hand-authored graph in this project
  already uses — conjunction ("put X and Y on the tray") and exclusion
  ("do not move Z" / "keep Z upright"). Object phrases resolve against a
  caller-supplied closed vocabulary (the objects that actually exist in that
  scene), not open vocabulary. An unrecognized clause raises instead of
  being silently dropped — silently ignoring a "do not move X" clause would
  itself be exactly the kind of intent violation this project exists to
  catch. Verified three ways: reproduces all three existing hand-authored
  graphs (canonical/replicacad/replicacad-humanoid) from their own
  instruction text; correctly parses held-out paraphrases never used to
  write the grammar (different verb, negation form, conjunction style,
  clause order, Oxford comma); correctly parses a held-out composition
  (objects recombined into a new sentence never written anywhere in this
  project). Wired into `tidy_up_env.py` for real — its `goal_graph` is now
  `parse_instruction(...)` output, not `canonical_example()` directly (which
  remains only as the parser's hand-authored reference/ground truth).
- **Reason:** Second stage of the build-up order in
  `docs/00-project-overview.md` — "parse an actual instruction sentence into
  the goal graph, instead of writing one by hand" — deliberately built and
  verified before adding vision or learning, so a failure is traceable to
  one new capability, not several.
- **Consequences:** Goal/constraint `id` strings are now generated
  (`place_<object_id>`, `dont_move_<object_id>`, etc.) rather than
  hand-chosen, which is why `tests/drafts/test_tidy_up_env.py`'s asserted
  ids changed (`place_red_mug`/`place_blue_bowl`, not `place_mug`/
  `place_bowl`) — cosmetic, nothing reads these ids besides dict keys and a
  guard-block message. Only `tidy_up_env.py` was switched over; the other
  three environments still build their graphs by hand — the parser already
  reproduces their instruction text exactly (see
  `tests/drafts/test_instruction_parser.py`), so switching them over is mechanical,
  not a further design question. Ordering/priority ("first... then...") and
  conditional goals are explicitly not implemented — no existing instruction
  uses them, and building grammar for them without a driving test case
  would be speculative per D-013's own scoping discipline.

## D-018: G1 placed in the real ReplicaCAD apartment — a second scene-builder bug found and fixed

- **Date:** 2026-07-30
- **Status:** Accepted
- **Decision:** Direct follow-up to "but this is not a humanoid robot":
  placed G1 (fixed-base, confirmed it cannot walk) into the same real
  apartment D-017 used, instead of Fetch. The obvious fix — catch
  `ReplicaCADSceneBuilder`'s fetch-only `NotImplementedError` — is wrong:
  the rearrange scene builder places objects in two passes (temporary
  pose+1000m-up, then real final pose), and the fetch-only check sits
  *between* them. Catching the exception skips the second pass, leaving
  every object floating at z≈1000 — found by inspecting actual object
  positions, not assumed. Real fix: temporarily present as `"fetch"` (plus
  alias a `"rest"` keyframe) so the builder completes its own correct
  logic, then set G1's real pose afterward. Also didn't assume a base
  position was reachable — raycast-checked several candidates first (same
  technique as D-017's path planner) before picking one with real open
  clearance. Same H2/H3 results as every other variant once placement was
  correct.
- **Reason:** Answering "is this genuinely embodiment-agnostic" requires
  actually trying a humanoid in the hardest environment tried so far, not
  just the two where we'd already worked out the friction points.
- **Consequences:** `ReplicaCADSceneBuilder`-based scenes have a real,
  non-obvious constraint: any robot besides `fetch` needs this same
  fetch-impersonation workaround, not a simple exception handler. Worth
  knowing before anyone else hits the same z≈1000 floating-object surprise.
  **Correction (D-020, 2026-07-31):** this decision's "same H2/H3 results"
  claim was only ever checked at seed=0. G1's hardcoded base pose and camera
  are calibrated for that one apartment layout specifically —
  `ReplicaCADSceneBuilder` loads a different room per seed, and other seeds
  place G1 nowhere near the relevant objects. Not caught until D-020's
  vision work rendered and looked at other seeds.
  **Fixed in D-021 (2026-07-31):** scene layout is now pinned regardless of
  seed; object placement and G1's reachability are confirmed identical
  across seeds by a regression test. D-020's separate rendering-state
  finding (see D-021) is unrelated to this and still open.
  Full detail in `spikes/task_schema_draft/README.md` "G1 in the real
  apartment."

## D-017: Real ReplicaCAD scene integration — needed real path planning, not a scene swap

- **Date:** 2026-07-30
- **Status:** Accepted
- **Decision:** Per direct request to prefer established environments over
  hand-built ones, rebuilt TidyUp on ManiSkill3's own `ReplicaCADSetTableTrain`
  scene builder — a real furnished apartment (104 actors, inspected directly)
  with real YCB objects, using the `fetch` mobile robot (the only supported
  option; `ReplicaCADSetTableTrain` initialization explicitly rejects
  `panda`). Found this scene's active objects are scattered across the whole
  apartment (rooms 1-2+ meters apart), so navigation — not just reach — is
  required. A naive point-and-drive controller got physically stuck on a
  real wall (confirmed via `PhysxCpuSystem.raycast`, not assumed). Built
  `navigation.py`: an occupancy grid from SAPIEN's own raycast API (no new
  dependency) plus Dijkstra shortest-path — deliberately not Habitat's
  bundled `.navmesh` files, which need `habitat-sim` and carry the same
  unverified-on-Apple-Silicon risk that `mplib` already cost us (D-011).
  Same qualitative H2/H3 results as the panda/humanoid variants once
  navigation worked.
- **Reason:** Established scenes solve calibration pain (footprints,
  settling) but don't remove the need to actually validate them — this
  scene's real complexity (multi-room scatter, real walls) was discovered
  empirically, not assumed away.
- **Consequences:** "Use an established environment" traded hand-placement
  calibration work for real path-planning work — a different kind of
  integration cost, not a free lunch. The occupancy grid's safety margin
  (0.2m) was tuned empirically after 0.3m (Fetch's actual base radius) sealed
  every doorway in the discretized grid; this margin is scene-specific, not
  a general constant. Full detail in `spikes/task_schema_draft/README.md`
  "ReplicaCAD embodiment."

## D-016: Task schema confirmed embodiment-agnostic — humanoid variant of TidyUp

- **Date:** 2026-07-29
- **Status:** Accepted (as a toy-scale demonstration, not a research result)
- **Decision:** Built `tidy_up_env_humanoid.py` / `policy_baselines_humanoid.py`
  — the same scene, goals, interventions, policies, and metrics as the panda
  version, on a Unitree G1 upper body instead. Same qualitative results
  (D-014/D-015's H2/H3 findings reproduce exactly). Confirms `goal_graph.py`,
  `oracle_feasibility.py`, and `intent_guard.py` are genuinely
  embodiment-agnostic. Required two adaptations: hand-calibrated joint-space
  reach configs (this G1 agent class has no Cartesian controller — checked
  directly, not just assumed), and a fix for a real settling bug (objects
  spawned above the kitchen counter's actual surface height tripped
  `dont_move_glass` from settling alone, before any policy acted) plus a
  counter-footprint asymmetry (x=-0.15 fell through empty space; x=+0.15 did
  not) — full detail in `spikes/task_schema_draft/README.md`.
- **Reason:** Requested directly, to confirm the schema logic isn't
  accidentally coupled to the panda arm before either contributor invests
  more in it.
- **Consequences:** Strengthens the case that D-013's schema draft is sound
  independent of embodiment choice. Still toy-scale, still needs teammate
  review — this doesn't change that. The joint-space reach calibration is
  hand-tuned for this exact scene layout, not a reusable IK solution; a real
  humanoid manipulation layer (per docs/07's "Strategy adaptation") is
  separate, later work.

## D-015: First runnable H3 test — intent guard blocks a constraint violation at zero recall cost (toy scale)

- **Date:** 2026-07-29
- **Status:** Accepted (as a toy-scale demonstration, not a research result)
- **Decision:** Built `intent_guard.validate_action()` (rejects an action
  targeting a `never_move`-constrained object unless a real goal requires
  it) and `naive_substitution_policy` (the "invalid agent" from docs/01:
  substitutes the glass for the destroyed bowl instead of accepting
  infeasibility). Unguarded: 1/2 goals, `dont_move_glass` violated.
  Guarded: 1/2 goals (identical), violation prevented — the substitution
  never earned goal credit either way, so blocking it was free here.
- **Reason:** First end-to-end test of H3, using the same infrastructure
  (goal graph, constraints, oracle checks) as D-014's H2 test.
- **Consequences:** This only demonstrates the easy case — zero-cost
  blocking. It does not test R-010's harder concern (a guard trivially
  avoiding violations by blocking *legitimate* actions, trading real recall
  for safety), which needs a scenario where guard precision is genuinely in
  tension with completing a real goal. Not built yet — a natural next step
  once this schema gets teammate review.

## D-014: First runnable H2 test — feasibility-aware policy beats static policy (toy scale)

- **Date:** 2026-07-29
- **Status:** Accepted (as a toy-scale demonstration, not a research result)
- **Decision:** Built `policy_baselines.py`: a `static_policy` (attempts
  goals in order regardless of feasibility) vs a `feasibility_aware_policy`
  (checks `goal_feasible()` before committing to the physical reach).
  Result after `bowl_destroyed`: both achieve 1/2 goals, but static wastes
  25 steps reaching for the now-destroyed bowl while feasibility-aware
  skips it (0 wasted steps, half the total steps). With no intervention,
  both achieve 2/2 with zero waste. Also fixed a real float32/float64
  boundary bug found while building this: `goal_achieved()`'s tray-height
  check rejected a real teleport-onto-tray placement because dz computed to
  -1.1e-10 instead of exactly 0.
- **Reason:** This is the first end-to-end demonstration of H2 (docs/01) —
  everything before this was schema/simulator infrastructure; this is the
  first time the actual research claim has been tested, even at toy scale.
- **Consequences:** This is existence-only feasibility (a direct privileged-
  state query), not learned feasibility, and "wasted steps" is a simplified
  cost proxy, not a reward/regret formulation — don't cite this as
  validating H2 in any general sense. It does validate that the schema +
  oracle + simulator plumbing built in D-013 is wired correctly enough to
  run a real comparison, which is what it was for.

## D-013: Draft task schema + intervention set, for review — not a commitment

- **Date:** 2026-07-29
- **Status:** Proposed (needs teammate review before "Accepted" — this is
  the "Shared" task-family/intervention-set item, not a unilateral call)
- **Decision:** Built a concrete, tested, runnable draft of docs/04's task
  schema (`spikes/task_schema_draft/`) around the project's own worked
  example from docs/01: "Put the red mug and blue bowl on the tray, keep the
  medicine upright, and do not move the glass." Includes a `GoalGraph` data
  model (goals/priorities/dependencies/constraints), pure-function oracle
  feasibility + constraint-violation checking (existence-based, never
  attempted-motion-based, per docs/04's own caution against that), and a
  ManiSkill3 scene wiring it to real privileged state — with one irreversible
  intervention (bowl destroyed) matched against one reversible/temporary
  control (a distractor object that appears and disappears), per docs/04's
  explicit requirement to include matched pairs.
- **Reason:** This was the single biggest bottleneck blocking further
  progress on both Person A's and Person B's tracks (STATUS.md). A concrete,
  runnable draft is easier to react to and critique than more prose in
  docs/04.
- **Consequences:** Not yet covered: language (deliberately Person A's
  territory), priorities/dependencies exercised by an actual example, actual
  goal-completion detection (vs. feasibility), held-out paraphrases, and the
  other four candidate intervention types (container broken, route
  permanently blocked, tool consumed, resource contention). See
  `spikes/task_schema_draft/README.md` "What this deliberately doesn't cover
  yet." Needs review with your teammate before anything here is treated as
  settled.

## D-012: Spike code made device-agnostic; found gotchas addressed, not just documented

- **Date:** 2026-07-28
- **Status:** Accepted
- **Decision:** Replaced hardcoded `sim_backend="cpu"` everywhere in
  `spikes/maniskill_humanoid_spike/` with `device_utils.resolve_sim_backend()`,
  which checks `torch.cuda.is_available()` directly — unlike ManiSkill3's own
  `sim_backend="auto"`, which only branches on `num_envs` and never checks
  CUDA availability. Also fixed the push force-application code in
  `humanoid_stand_spike.py` to branch between the CPU per-body API and the
  GPU batched-tensor API (it previously only worked on CPU). Object
  add/remove (`object_intervention_spike.py`) is a genuine GPU-sim
  limitation, not a gap in our code — added an explicit `RuntimeError` guard
  there instead of pretending it's portable.
- **Reason:** Requested directly — run on CUDA if available, fall back to
  CPU, and the code should work unmodified on whichever machine it lands on
  next (this dev machine, a teammate's machine, or a cloud GPU box).
- **Consequences:** The CPU path is fully re-verified (identical spike
  results before/after this refactor). The GPU path is written correctly by
  inspection and follows the same pattern ManiSkill3's own `Actor.apply_force`
  uses internally, but is **untested** — this dev machine has no CUDA. Verify
  on a CUDA machine before trusting it for anything real.

## D-011: ManiSkill3 RGB-D and basic manipulation confirmed; canned motion planning is not portable here

- **Date:** 2026-07-28
- **Status:** Accepted
- **Decision:** Extended the spike again (`manipulation_skill_spike.py`) to
  test RGB-D observations and the "reusable reach/grasp" requirement.
  RGB-D (`obs_mode="rgbd"`) works cleanly on `PickCube-v1`. ManiSkill3's
  shipped motion-planning solutions depend on `mplib`, which fails to build
  on this machine (Apple Silicon macOS, pins `libclang==11.0.1`, no matching
  wheel). Worked around it using the built-in `pd_ee_delta_pos` Cartesian
  controller (IK via `pinocchio`, installable here as `pin`) with a simple
  hand-scripted waypoint sequence — picked up and lifted a cube 5/5 times
  across seeds 0-4.
- **Reason:** These were the last two untested rows in the selection
  requirements table besides language (not a simulator capability) and
  Isaac Lab comparison.
- **Consequences:** ManiSkill3 now clears every testable requirement.
  `mplib`/collision-aware motion planning is a known platform gap on Apple
  Silicon dev machines specifically — if collision-aware planning turns out
  to matter later, budget time to resolve the `mplib` build or use a
  different planner, rather than assuming the shipped examples work
  out of the box. I-003 stays open only pending an Isaac Lab spike.

## D-010: ManiSkill3 object-level interventions confirmed working

- **Date:** 2026-07-28
- **Status:** Accepted
- **Decision:** Extended the spike (`object_intervention_spike.py`) to test
  the requirement that actually gates the simulator decision: can the
  simulator realize `WorldIntervention`-style object/scene changes, not just
  a physical push? Confirmed on ManiSkill3: an object can be genuinely
  removed from the live physics scene mid-episode, and new geometry (a
  blocking obstacle) can be added to an already-built scene mid-episode —
  both deterministic given a seed. Also found a real gotcha: the high-level
  `Actor` Python wrapper goes stale after removal (keeps returning
  pre-removal pose/state instead of erroring), so any oracle/eval code must
  track object existence itself rather than re-querying the wrapper.
- **Reason:** Standing balance (D-009) turned out not to be the hard
  question — object-level intervention support was the actual unknown that
  mattered, per docs/04-benchmark-environment.md's "Candidate irreversible
  changes" and the `WorldIntervention` API sketch.
- **Consequences:** ManiSkill3 now clears every requirement tested so far
  (humanoid support, seeding, privileged state, object-level interventions).
  Still open before I-003 can close: RGB/language integration, the reusable
  skill library, and an equivalent Isaac Lab spike for comparison. See
  `spikes/maniskill_humanoid_spike/README.md` for full results.

## D-009: ManiSkill3 humanoid spike — findings, not a simulator selection

- **Date:** 2026-07-28
- **Status:** Accepted
- **Decision:** Ran the Phase 0 simulator spike D-006 calls for, against
  ManiSkill3 specifically: `spikes/maniskill_humanoid_spike/` (deliberately
  outside `src/`, since D-006 says not to commit simulator-specific
  architecture yet). Confirms humanoid asset support (Unitree G1 bundled, H1
  one download away), exact deterministic seeding of a scripted event, and
  privileged-state access. Does **not** confirm RGB/language integration or
  the skill library — object-level intervention support was confirmed
  separately, see D-010.
- **Reason:** Needed concrete evidence before the simulator decision could be
  anything but a guess; D-006 explicitly requires this spike step.
- **Consequences:** ManiSkill3 remains a strong candidate, not a final
  selection — I-003 stays open until Isaac Lab gets an equivalent spike and
  the remaining untested requirements (RGB, language, skills) are checked.
  Also recorded: no CUDA on the primary dev machine (Apple M4 Max), so
  SAPIEN's GPU-vectorized backend is unavailable there; CPU backend is fine
  for single-env dev (~450–600 steps/sec) but large-scale parallel RL
  training will need a CUDA machine regardless of which simulator is chosen.
  See `spikes/maniskill_humanoid_spike/README.md` for full results.

## D-008: Two-person ownership with shared benchmark first

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Both contributors build the benchmark and contracts first.
  Person A then leads representation/language/feasibility; Person B leads
  policy/humanoid execution. Integration and final evaluation remain shared.
- **Reason:** This balances specialization with the need to test the research
  question at the perception-policy boundary and avoids late integration.
- **Consequences:** Person A develops against recorded trajectories, Person B
  against oracle beliefs, interfaces are versioned, and roadmap phases contain
  explicit integration gates.

## D-007: Simulated humanoid is the required target embodiment

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Keep feasibility and intent reasoning embodiment-agnostic, but
  require final evaluation on a simulated humanoid using a stable skill interface.
- **Reason:** The project is intended to apply to humanoids without conflating
  high-level strategy adaptation with learning whole-body control from scratch.
- **Consequences:** Simulator selection must support humanoids; Phase 0 validates
  an asset and low-level skills; results separate skill failure from incorrect
  infeasibility; simpler embodiments may be used only as intermediate testbeds.

## D-004: Feasibility-aware vision-language RL research direction

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Study whether a vision-language RL agent using self-supervised
  visual representations can infer goal feasibility after unforeseen,
  irreversible world changes and adapt without violating the original intent.
- **Reason:** This is the project's new primary research question.
- **Consequences:** The previous humanoid failure-monitor and recovery-skill
  architecture is superseded. Environment, modules, metrics, roadmap, and
  diagram must support language goals, feasibility, and intent constraints.

## D-005: Operational definition of intent

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Represent original intent as atomic goals, dependencies,
  priorities, hard constraints, and explicit substitution/equivalence rules.
- **Reason:** “Intent” must be machine-checkable for training and evaluation.
- **Consequences:** Claims are limited to this schema and must not imply general
  human-intent understanding.

## D-006: Simulator remains undecided

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Evaluate candidate object-centric visual environments before
  selecting a primary humanoid-capable simulator.
- **Reason:** ManiSkill was chosen for the old humanoid-control question; the new
  study prioritizes intervention control, language, and oracle feasibility.
- **Consequences:** Phase 0 includes a simulator spike. No simulator-specific
  architecture should be committed before it passes the selection criteria.

## D-001: Simulation-only scope

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Develop and evaluate v1 in simulation.
- **Reason:** Enables reproducible interventions and privileged oracle labels.
- **Consequences:** Claims do not extend to real robots without further evidence.

## D-002: ManiSkill as primary simulator

- **Date:** 2026-07-24
- **Status:** Superseded by D-006
- **Decision:** Originally selected ManiSkill for humanoid recovery experiments.
- **Reason:** No longer aligned with the revised question by default.
- **Consequences:** ManiSkill is now one candidate rather than a commitment.

## D-003: Separate stable docs from live tracking

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Keep stable design in `docs/` and live notes in `ai-notes/`.
- **Reason:** They have different audiences and update rhythms.
- **Consequences:** Keep cross-links and status consistent.

## Template

```text
## D-NNN: Short title
- Date / Status / Decision / Reason / Consequences
```
