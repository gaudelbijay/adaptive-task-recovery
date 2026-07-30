# Decisions

Lightweight architecture decision log. Stable research design is in `docs/`.

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
