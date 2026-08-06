# Task schema + intervention set — evidence for `src/atr/`'s promoted core

**What this is:** the environments, policies, language/vision/
representation layers, and learned policy that were built to test D-013's
task schema. **The schema itself — `Goal`/`Constraint`/`GoalGraph`, oracle
feasibility, the intent guard — has been reviewed and promoted to
[`src/atr/`](../../src/atr/) (D-037, 2026-08-02).** Read
[`ai-notes/review-request-task-schema.md`](../../ai-notes/review-request-task-schema.md)'s
status banner before treating that as equivalent to independent review —
it was self-resolved by the project owner, not evaluated by the teammate
it was written for. Everything in *this* directory remains what it always
was: evidence for or against that schema, not part of it, and none of it
has made its own case for promotion yet.

## The idea in one sentence

Take the project's own worked example (docs/01
"Example"): *"Put the red mug and blue bowl on the tray, keep the medicine
upright, and do not move the glass"* — and build the smallest possible
runnable version of it, so the goal-graph schema and the oracle-feasibility
logic in docs/04-benchmark-environment.md aren't just prose anymore.

## What's here

| File | What it is |
|---|---|
| [`../../src/atr/language/goal_graph.py`](../../src/atr/language/goal_graph.py) *(promoted, D-037)* | `Goal`, `Constraint`, `GoalGraph` dataclasses matching docs/04's "atomic goals, priorities, dependencies, and hard constraints." `canonical_example()` builds the docs/01 instruction as data; `dependent_goals_example()` exercises `Goal.depends_on`. |
| [`../../src/atr/feasibility/oracle.py`](../../src/atr/feasibility/oracle.py) *(promoted, D-037)* | Pure functions: `goal_feasible()` (exists-based, never attempted-motion-based — see "Humanoid validity requirements" in docs/04), `goal_dependencies_satisfied()` (D-037), and `constraint_violated()` (position-drift / orientation checks), plus `evaluate_goal_graph()` combining the feasibility/violation checks. No simulator dependency — testable in isolation. |
| [`../../src/atr/envs/tidy_up_env.py`](../../src/atr/envs/tidy_up_env.py) *(promoted, D-045)* | A ManiSkill3 scene (5 objects: red_mug, blue_bowl, tray, medicine_bottle, glass + an idle `panda` arm) wiring the schema above to real privileged state. Registered as `TidyUp-v1` (was `TidyUpTaskSchemaDraft-v1` before promotion). |
| [`../../src/atr/envs/tidy_up_policies.py`](../../src/atr/envs/tidy_up_policies.py) *(promoted, D-046)* | `static_policy()` vs `feasibility_aware_policy()` — the first runnable test of H2 (docs/01): does checking feasibility before acting beat a policy that doesn't? Also `naive_substitution_policy()`, used by the intent guard test below. Supplies `attempt_goal()` (real arm motion) and tray geometry for the canonical env; the shared decision logic lives in `src/atr/policies/baselines.py` (D-040). |
| [`../../src/atr/constraints/intent_guard.py`](../../src/atr/constraints/intent_guard.py) *(promoted, D-037)* | `validate_action()` — the first runnable test of H3 (docs/01): does rejecting an unauthorized action before execution reduce constraint violations? |
| [`../../src/atr/envs/tidy_up_env_humanoid.py`](../../src/atr/envs/tidy_up_env_humanoid.py) *(promoted, D-047)* | The same scene and interventions, on a Unitree G1 upper body instead of the panda arm — proves `goal_graph`/`oracle_feasibility`/`intent_guard` are genuinely embodiment-agnostic, not accidentally panda-specific. Registered as `TidyUp-Humanoid-v1` (was `TidyUpTaskSchemaDraft-Humanoid-v1`). |
| [`../../src/atr/envs/tidy_up_humanoid_policies.py`](../../src/atr/envs/tidy_up_humanoid_policies.py) *(promoted, D-047)* | The same `static_policy` / `feasibility_aware_policy` / `naive_substitution_policy`, adapted to the humanoid's joint-space-only control (see below). |
| [`../../src/atr/envs/tidy_up_env_replicacad.py`](../../src/atr/envs/tidy_up_env_replicacad.py) *(promoted, D-048)* | The same goals/interventions on a **real** furnished apartment (ManiSkill3's own `ReplicaCADSetTableTrain` scene builder, real YCB objects) with a mobile Fetch robot, instead of a hand-built scene. Registered as `TidyUp-ReplicaCAD-v1` (was `TidyUpTaskSchemaDraft-ReplicaCAD-v1`). |
| [`../../src/atr/envs/navigation.py`](../../src/atr/envs/navigation.py) *(promoted, D-048)* | A grid + Dijkstra path planner, built because a naive point-and-drive controller got physically stuck on a real wall in this scene (see "ReplicaCAD embodiment" below). |
| [`../../src/atr/envs/tidy_up_replicacad_policies.py`](../../src/atr/envs/tidy_up_replicacad_policies.py) *(promoted, D-048)* | The same three policies, navigating (not just reaching) to each goal. |
| [`../../src/atr/envs/tidy_up_env_replicacad_humanoid.py`](../../src/atr/envs/tidy_up_env_replicacad_humanoid.py) *(promoted, D-049)* | G1 (fixed-base, no navigation) placed in the same real apartment instead of Fetch — the direct "but this is not a humanoid robot" answer. Registered as `TidyUp-ReplicaCAD-Humanoid-v1` (was `TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1`). |
| [`../../src/atr/envs/tidy_up_replicacad_humanoid_policies.py`](../../src/atr/envs/tidy_up_replicacad_humanoid_policies.py) *(promoted, D-049)* | The same three policies, arm-reach only (no navigation — G1 can't move its base). |
| [`../../src/atr/language/instruction_parser.py`](../../src/atr/language/instruction_parser.py) *(promoted, D-038)* | `parse_instruction()` — turns an instruction sentence into a `GoalGraph`, instead of writing one by hand. Controlled grammar, closed object vocabulary. Wired into `tidy_up_env.py`. See "Language layer" below. |
| [`../../src/atr/feasibility/clip_feasibility.py`](../../src/atr/feasibility/clip_feasibility.py) *(promoted, D-039)* | `visual_object_exists()` — judges object presence from a rendered camera frame using zero-shot CLIP, instead of reading privileged state. Calibrated against `tidy_up_env_replicacad_humanoid.py`. See "Vision layer" below. |
| [`dinov2_probe.py`](dinov2_probe.py) | `dinov2_embed()` + `fit_and_evaluate_probe()` — a self-supervised (no text/labels) embedding plus a linear probe, instead of CLIP's language-aligned zero-shot judgment. `fit_probe()`/`run_end_to_end_episode_dinov2()`/`collect_arm_occluded_examples()` (D-054/D-055) wire it into a real live decision loop and close the robustness gap doing so first surfaced. Still not promoted. See "Self-supervised representation layer" below. |
| [`../../src/atr/envs/capture_episode_subprocess.py`](../../src/atr/envs/capture_episode_subprocess.py) *(promoted, D-052)* | One-shot render capture, run only as a subprocess — works around D-022 (a confirmed upstream ManiSkill3 rendering bug) by making every labeled example the OS's "first" render-producing reset. |
| [`rl_policy.py`](rl_policy.py) | `train_q_table_canonical()` + `learned_policy()` — tabular Q-learning that discovers "attempt iff feasible" from reward, instead of `feasibility_aware_policy`'s hard-coded rule. Thin wrapper over [`../../src/atr/policies/q_learning.py`](../../src/atr/policies/q_learning.py) since D-041. See "Learned policy" below. |
| [`../../src/atr/control/ik_solver.py`](../../src/atr/control/ik_solver.py) *(promoted, D-051)* | `solve_right_arm_ik()` + `best_reachable_distance()` — a real analytic-Jacobian IK solver on `pinocchio`, built to retry D-024's grasp confirmation properly. See "Real IK retry" below. |
| [`../../src/atr/pipeline.py`](../../src/atr/pipeline.py) *(promoted, D-050)* | `run_end_to_end_episode()` — language parsing, real vision-based feasibility, and a learned policy, combined into one real episode. See "Everything combined" below. |

## The two interventions (matched, per docs/04)

docs/04 explicitly warns: *"Include matched reversible and temporary
changes. Otherwise the model may learn that every detected change implies
abandonment."* So this ships two, not one:

- **`bowl_destroyed`** (irreversible): removes `blue_bowl` mid-episode via
  the same `remove_from_scene()` mechanic validated in
  `spikes/maniskill_humanoid_spike/`. Verified result: `place_bowl` goal
  flips to infeasible, `place_mug` stays feasible, no constraint is
  violated — exactly the docs/01 worked example.
- **`temporary_obstacle`** (reversible/matched control): spawns a distractor
  object near the tray, then removes it again a few steps later. Verified
  result: goal feasibility and constraint violations never change — a
  detectable world event that correctly implies *nothing* about
  infeasibility. Without this matched control, an oracle/model trained only
  on the irreversible case could over-generalize "any detected change ->
  abandon the goal."

Both are deterministic given a seed (same pattern as the earlier spikes).

## Verified findings (2026-07-29)

- The exact docs/01 scenario runs and produces the exact expected labels:
  bowl destroyed -> `place_bowl` infeasible, `place_mug` unaffected, zero
  constraint violations.
- Constraint violation detection was checked for true positives too, not
  just the not-violated nominal case: manually displacing the glass 10cm
  correctly flips `dont_move_glass` to `True`; manually tipping the
  medicine bottle 90° correctly flips `keep_medicine_upright` to `True`.
- A destroyed object never counts as a constraint violation on itself (that
  situation is a feasibility question, not a "did it move / tip over"
  question) — deliberate, tested design choice, not an oversight.
- CPU sim only, same reason as `object_intervention_spike.py`: object
  add/remove is unsupported under GPU-batched sim. `_initialize_episode`
  raises a clear `RuntimeError` if that's violated.

## First H2 result (2026-07-29)

docs/01's H2 hypothesis: *"conditioning strategy selection on per-goal
feasibility estimates outperforms a static language-conditioned policy after
irreversible changes."* `atr.envs.tidy_up_policies` (promoted, D-046; was `policy_baselines.py`) runs the smallest possible
version of that test: both policies attempt `place_mug` then `place_bowl` in
order, using real arm motion for the reach phase (see the module's scope
note on what's abstracted — the grasp mechanic itself, already validated
separately in `maniskill_humanoid_spike/manipulation_skill_spike.py`). After
`bowl_destroyed` fires before the bowl is attempted:

| Policy | Goals achieved | Total steps | Wasted steps |
|---|---|---|---|
| `static_policy` (no feasibility check) | 1/2 | 50 | 25 |
| `feasibility_aware_policy` (checks first) | 1/2 | 25 | 0 |

Same outcome, half the effort, zero wasted attempts on the now-infeasible
goal — the feasibility-aware policy checks `goal_feasible()` (a privileged-
state query, ~zero cost) before committing to the physical reach, and skips
`place_bowl` immediately instead of reaching for an object that's gone.
With no intervention, both policies achieve 2/2 goals with zero waste (see
`tests/drafts/test_tidy_up_policies.py`).

This is a toy-scale, existence-only version of H2, not a publishable result:
no learned feasibility estimation (the check is a direct privileged-state
query), no language, and "wasted steps" is a simplified proxy for cost, not
a full reward/regret formulation. But it's the first time any part of this
project's central research question has been demonstrated end to end,
rather than argued for in prose.

## First H3 result (2026-07-29)

H3: *"explicit goal/constraint checking reduces semantic and constraint
violations with an acceptable trade-off in achievable-goal recall."*
`naive_substitution_policy()` in `atr.envs.tidy_up_policies` is the "invalid
agent" from docs/01's own worked example: when `place_bowl` turns out
infeasible, instead of accepting that, it substitutes the glass onto the
tray — which never legitimately satisfies the goal (the bowl still doesn't
exist) and violates `dont_move_glass`. `intent_guard.validate_action()`
checks a candidate action against the goal graph's hard constraints before
it's executed:

| | Goals achieved | `dont_move_glass` violated | Substitution attempted |
|---|---|---|---|
| Unguarded | 1/2 | **Yes** | Yes |
| Guarded | 1/2 | **No** | Blocked before execution |

Same recall, zero violations, once the guard is in place. Notably, this is
the *easy* case for H3 — the substitution never earned any goal credit
either way, so blocking it costs nothing. It does not test the harder,
more interesting failure mode named in R-010
(`ai-notes/issues_and_risks.md`): a guard that trivially avoids violations
by blocking *legitimate* actions too, trading real recall for safety. That
needs a scenario where the guard's precision is actually in tension with
completing a real goal — not built yet.

## Humanoid embodiment (2026-07-29)

Same scene, same goals, same interventions, same policies and metrics —
`tidy_up_env_humanoid.py` swaps the panda arm for a Unitree G1 upper body
(`unitree_g1_simplified_upper_body_with_head_camera`, the same agent class
ManiSkill3's own `UnitreeG1PlaceAppleInBowl-v1` example uses) on a kitchen
counter. Ran the exact same policy comparisons and got the exact same
qualitative results as the panda version — 1/2 goals achieved either way,
`feasibility_aware` at 0 wasted steps vs `static`'s 25, guard blocks the
substitution at zero recall cost — confirming `goal_graph.py` /
`oracle_feasibility.py` / `intent_guard.py` really are embodiment-agnostic,
not accidentally coupled to the panda arm.

Two real things had to change, both documented in
`tidy_up_env_humanoid.py`'s module docstring:

- **No Cartesian controller.** This G1 agent class only exposes
  `pd_joint_pos`/`pd_joint_delta_pos` — no `pd_ee_delta_pos` like Panda has.
  Rather than building one (a bigger task — subclassing the agent to add a
  `PDEEPosControllerConfig` for the right arm), the "reach" phase uses two
  hand-calibrated joint configurations (`_REACH_CONFIGS`), found by sweeping
  shoulder/elbow angles and reading off `agent.right_tcp.pose` empirically.
  Less general than real IK, but it's real arm motion consuming real steps,
  which is what the "wasted effort" metric actually needs.
- **A real settling bug, found and fixed.** Objects spawned at an assumed
  counter height dropped a few centimeters onto the kitchen counter's real
  surface in the first 1-3 steps — enough to trip `dont_move_glass` on its
  own, before any policy did anything. Fixed by freezing the
  never-move/upright baseline a few steps after reset instead of at the
  instant of spawn (`tidy_up_env_humanoid.py`'s `evaluate()`), plus an
  explicit settle period in `naive_substitution_policy` before it captures
  its own baseline (it reads state directly, bypassing `evaluate()`'s fix).
  Also found: the counter's collision footprint isn't symmetric around
  x=0 — an object at x=-0.15 fell straight through empty space while
  x=+0.15 rested fine. Objects are now placed only on the proven side.

## ReplicaCAD embodiment (2026-07-29/30)

Requested directly: prefer established environments over hand-built ones
where they exist. `tidy_up_env_replicacad.py` swaps the hand-built scene for
ManiSkill3's own `ReplicaCADSetTableTrain` scene builder — a real furnished
apartment (104 actors, inspected directly) with real YCB objects from
Habitat's rearrangement dataset. `master_chef_can`, `bowl`, `potted_meat_can`,
`cracker_box` are genuine YCB models, not primitives; `env-0_024_bowl-4` is
an actual bowl mesh with real geometry and mass. Requires downloading
ReplicaCAD (~1.6GB), ReplicaCADRearrange (~2.8GB), and the YCB object set.

**This was not a drop-in scene swap, and that's worth knowing before reaching
for it again:**

- **These scenes need a mobile robot, not a fixed arm.** `ReplicaCADSetTableTrain`
  only supports `fetch`, and its objects are scattered across the *entire
  apartment* (checked real positions — rooms 1-2+ meters apart), not
  clustered on one table like the panda/humanoid scenes. So "attempt a
  goal" now genuinely requires navigation, not just a reach.
- **A naive point-and-drive controller physically got stuck on a real
  wall.** Confirmed directly, not assumed: a raycast at the exact stuck
  position hit a `PhysxRigidStaticComponent` 0.29m away in the direction of
  travel. Fixed with `navigation.py`: a 2D occupancy grid built from
  SAPIEN's own `PhysxCpuSystem.raycast` (no new dependency) plus Dijkstra
  shortest-path, then the same proportional controller follows the
  resulting waypoints instead of driving straight at the target.
  Deliberately not Habitat's own `.navmesh` files (bundled with the
  dataset) — parsing those needs `habitat-sim`, a heavy C++ package with
  the same unverified-on-Apple-Silicon risk profile that burned us on
  `mplib` (D-011).
- **The occupancy grid's safety margin needed real tuning, not a
  guess.** A margin equal to Fetch's actual base radius (0.3m) left every
  doorway in this scene fully sealed in the discretized grid — no path
  existed at all between rooms. 0.2m was the largest margin that still
  found a path; verified by testing several values, not assumed safe.
- Same abstractions as the other variants otherwise: placement is a
  teleport-on-success, and interventions/oracle/intent-guard logic is
  unchanged (only the object alias map and tray position differ).

Net result: same qualitative findings as the panda/humanoid versions (1/2
goals achieved either way; feasibility-aware wastes 0 steps vs. static's
~250; the intent guard blocks the substitution at zero recall cost) — now
demonstrated on a real, unmodified, professionally-built scene with a real
mobile robot, confirming the schema/oracle/guard layer transfers without
change to genuinely different environment complexity, not just different
robot geometry.

## G1 in the real apartment (2026-07-30)

The Fetch demo above answers "does the schema work in a real environment,"
but drew the follow-up: *that's not a humanoid*. `tidy_up_env_replicacad_humanoid.py`
places G1 (fixed-base, legs locked — checked its class definition directly,
it physically cannot walk) into the *same* real apartment instead of Fetch.

Two real problems, found by testing, not by inspection:

- **The obvious fix (catch the fetch-only `NotImplementedError`) is wrong.**
  `ReplicaCADRearrangeSceneBuilder.initialize()` places objects in **two
  passes**: first at a temporary pose shifted 1000m up (to dodge leftover
  collision state from the previous episode), then — *after* the
  fetch-specific robot-teleport check that raises for anything else — at
  their real final pose. Catching the exception there skips the second
  pass entirely. Found this by inspecting actual object positions after
  using the "obvious" fix: every object was floating at z≈1000, not
  resting on furniture. **Real fix:** temporarily present as `"fetch"`
  (and alias a `"rest"` keyframe to G1's `"standing"` one, since the base
  class's fetch branch expects a keyframe G1 doesn't have) so the builder
  completes its own correct placement logic, then set G1's real pose
  afterward. Locked in as a regression test
  (`test_objects_are_placed_at_real_positions_not_floating`).
- **No base position was assumed reachable — checked first.** Raycast at
  candidate standing spots (same technique as `navigation.py`) found real
  nearby obstacles at several candidates before landing on `[0.55, 0.23]` as
  the most open one tried, with both target objects (`potted_meat_can`,
  `master_chef_can`) within the arm's reach envelope from there.

Because G1 can't navigate, goal/constraint roles are swapped from the Fetch
variant: the two objects near the chosen standing spot are the goals (one
gets destroyed by the intervention); the bowl and cracker box — genuinely
out of this fixed-base robot's reach — are constraint-only, monitored via
privileged state, never touched. Same result once placement was fixed: 1/2
goals either way, feasibility-aware wastes 0 steps vs. static's 25, guard
blocks the substitution at zero recall cost.

## Language layer (2026-07-30)

`instruction_parser.py`'s `parse_instruction(text, known_objects)` is stage 2 of the
build-up order in `docs/00-project-overview.md`: turn an instruction
sentence into a `GoalGraph`, instead of writing one by hand. Controlled
grammar, not open-ended NLU — covers exactly the two clause forms every
hand-authored graph in this project already uses:

- **Conjunction:** "put the X and Y on the tray" → one `on_tray` goal per
  object.
- **Exclusion:** "do not move Z" / "don't move Z" / "never move Z" /
  "leave Z alone" → `never_move`; "keep Z upright" → `maintain_orientation`.

Object phrases resolve against a caller-supplied closed vocabulary (the
objects that actually exist in the scene the instruction will run in), not
open vocabulary — matching this project's existing closed-world,
object-centric scope. An unrecognized clause raises `ValueError` rather than
being silently dropped: silently ignoring a "do not move X" clause would
itself be exactly the intent violation this project exists to catch.

Verified three ways (`tests/drafts/test_instruction_parser.py`):

- **Reproduces** all three existing hand-authored graphs (canonical,
  ReplicaCAD, ReplicaCAD-humanoid) from their own `instruction_text`.
- **Held-out paraphrases** — sentences never used to write the grammar:
  different verb ("place" vs "put"), different negation ("never move" /
  "leave alone" vs "do not move"), no-comma all-"and" phrasing, Oxford
  comma, reversed clause order.
- **Held-out composition** — a new instruction recombining objects across
  scenes that was never written as a `GoalGraph` anywhere in this project.

Wired into `tidy_up_env.py` for real: its `goal_graph` is now
`parse_instruction(CANONICAL_INSTRUCTION_TEXT, CANONICAL_OBJECTS)`, not
`canonical_example()` directly — confirmed live, not just in tests: the
bowl-destroyed intervention still correctly marks the parsed
`place_blue_bowl` goal infeasible while `place_red_mug` stays feasible.
`canonical_example()` itself is kept as the parser's hand-authored
reference, not deleted. The other three environments still build their
graphs by hand; the parser already reproduces their instruction text
exactly, so switching them over is mechanical, not a further design
question.

**Ordering/priority and conditional goals, added 2026-08-01 (D-026).**
"First put the mug on the tray, then put the bowl on the tray" assigns
sequential `Goal.priority` in order of appearance among order-marked
clauses; unmarked clauses keep priority=0, so nothing existing changed.
"If the blue bowl is destroyed, put the backup bowl on the tray instead"
sets a new field, `Goal.condition = (trigger_object_id, required_exists)`
— checked by `oracle_feasibility.py`'s `goal_feasible()` before the goal's
own target object even matters. Real design snag, found by testing: the
generic clause splitter breaks any comma right before a recognized verb
("put"), which is exactly the shape of "if X is Y, put Z on the tray" — so
conditional clauses get extracted in a separate pass *before* the generic
splitter runs on whatever's left, rather than being just another branch in
the normal clause classifier. `Goal.condition` is a schema change, and is
explicitly marked PROPOSED, not Accepted — same "needs review" status
D-013 itself has (see `ai-notes/review-request-task-schema.md`); adding it
doesn't get to unilaterally settle the exact question that review request
is asking your teammate to weigh in on.

## Vision layer (2026-07-31)

`clip_feasibility.py`'s `visual_object_exists(frame, object_id)` is stage 3 of the
build-up order in `docs/00-project-overview.md`: judge feasibility from a
rendered image instead of privileged state, starting with any working
pretrained model — zero-shot CLIP (`open_clip`, ViT-B-32, OpenAI weights),
no training or fine-tuning.

Getting this to work at all took four real findings, none assumed going in:

- **Whole-frame CLIP similarity is too weak a signal.** A global image/text
  score for "a photo of a blue bowl" barely moved when the bowl was actually
  removed (measured delta ~0.01, sometimes the wrong sign, across 20 seeds).
  The object is a small fraction of a cluttered frame — a known CLIP weak
  point. A tight crop around the object's known on-screen location (fixed
  camera → fixed crop, camera calibration, not a live 3D-position read)
  fixed this.
- **`tidy_up_env.py`'s objects are plain colored boxes**, not the objects
  they're named after (`build_box` primitives standing in for "mug" /
  "bowl"). Zero-shot CLIP correctly can't recognize "a blue bowl" in a
  picture of a blue cube, because there isn't one — a scene-realism
  mismatch, not a CLIP failure. Calibration moved to
  `tidy_up_env_replicacad_humanoid.py` instead, which has real
  photorealistic YCB-scanned objects.
- **A real, previously-latent bug:** removing an object in
  `_trigger_intervention()`'s `chef_can_destroyed` branch never called
  `self.scene.update_render()` — every existing consumer of this env reads
  privileged state, not pixels, so a stale render went unnoticed until this
  was the first code to actually look at a frame after a removal. Fixed by
  matching the `update_render()` call the `temporary_obstacle` branch
  already had.
- **A second real, previously-latent bug, found but not fixed:** G1's
  hardcoded base pose and camera in `tidy_up_env_replicacad_humanoid.py` are
  calibrated for exactly one apartment layout. `ReplicaCADSetTableTrain`
  loads a different room per seed — rendering seed=2 placed G1 next to a
  couch and a bicycle, nowhere near the cans. Every prior test of that env
  (D-018) only ever used seed=0, so this was never caught until vision work
  rendered and looked at other seeds. `tests/drafts/test_clip_feasibility.py` is
  deliberately seed=0-only because of this — see D-020 in
  `ai-notes/decisions.md`.

Generic object descriptions also measurably underperformed specific/iconic
ones — "a photo of a green can" gave a much weaker signal for the potted
meat can than "a photo of a Spam can". Both objects' final calibration:
a hand-picked crop plus prompt in `_OBJECT_VISUAL_CONFIG`, matching oracle
feasibility on all 4 cases tested (both objects, before and after the
intervention) at seed=0. **Not a general accuracy claim** — one scene
layout, four data points, an object detector for exactly two calibrated
objects, not a general one (raises rather than guessing for anything else).

## Scene-layout generalization fix (2026-07-31)

D-020's finding #4 (G1 placement only ever validated at seed=0) is fixed —
see D-021 in `ai-notes/decisions.md`. Root cause:
`ReplicaCADRearrangeSceneBuilder` samples both the apartment layout and
which YCB objects are actually placed (vs. hidden at z=-10000) from torch's
*global* RNG, independent of this project's own per-episode
`_episode_rng`. Confirmed broken in **both** real-scene envs, not just the
humanoid one — `tidy_up_env_replicacad.py`'s `env.reset(seed=2)` hid both
of its own goal objects outright. Fixed in both files by pinning
`build_config_idxs`/`init_config_idxs` to the config `reset(seed=0)`
happened to sample originally, and reseeding torch's global RNG
(`torch.manual_seed(0)`) immediately before both scene-construction calls.
`test_scene_layout_reproducible_across_seeds` (added to both envs' test
files) confirms all four target objects now land at byte-identical
positions across seeds {0, 2, 7/15, 42}.

**A separate, deeper issue turned up while verifying this — investigated
properly, confirmed as a known upstream bug, not fixable here** (D-022 in
`ai-notes/decisions.md`). Rendered frames for both real-scene envs can
desync from the actual scene (object positions stay correct; the image
doesn't — sometimes showing entirely different furniture) after roughly
the second render-producing reset within one Python process. Ruled out,
each tested directly rather than assumed: seed (identical `seed=0` config,
repeated, still degrades); reconfigure timing (forcing
`options={"reconfigure": True}` on every reset doesn't help);
`sapien.render.clear_cache()`; lighting values (`ambient_light` and
light-entity count are identical across instantiations). Reproduces on
**both** envs, ruling out anything specific to either one's own code.
Then checked whether this is a known upstream bug rather than stopping at
an educated guess: it is —
[haosulab/ManiSkill#1150](https://github.com/haosulab/ManiSkill/issues/1150),
open, unfixed, no maintainer workaround, matching this exactly (macOS-only,
specifically the YCB-object-loading environments, breaking after the
2nd/3rd reset). Not something to patch in this project. Both envs now
count render-producing resets and warn past the point actually verified
safe (`_render_producing_reset_count` in each file), citing the upstream
issue directly, converting a silent wrong answer into a loud one.
`tests/drafts/test_clip_feasibility.py` keeps exactly two render-producing resets and
both were visually re-verified against saved frames directly, not just
trusted from the CLIP score.

## Self-supervised representation layer (2026-08-01)

`dinov2_probe.py` is stage 4 of the build-up order in
`docs/00-project-overview.md`: swap in a representation learned from
unlabeled data, once stage 3 (any working pretrained model) works at all.
Deliberately a different kind of model from clip_feasibility.py's CLIP, not just a
bigger one — DINOv2 (`facebookresearch/dinov2`, ViT-S/14) is trained with
**no text or labels whatsoever**, purely self-supervised on images. It has
no built-in notion of "coffee can" or "exists" the way CLIP's zero-shot
prompting does; the only way to use it here is to fit a small linear probe
on labeled (embedding, exists) examples and check whether the two classes
are linearly separable in its feature space — the standard way
self-supervised representations get evaluated.

Result: a logistic-regression probe on 8 examples (master_chef_can, 4
present / 4 absent) reaches 100% leave-one-out cross-validation accuracy —
matching zero-shot CLIP's result on the same task, but from a
representation that was never told what any of these words mean.

D-022's confirmed upstream rendering bug caps safe render-producing resets
at roughly 2 per process — nowhere near enough to collect a probing
dataset. Worked around it rather than blocking on it:
`capture_episode_subprocess.py` captures exactly one labeled example and
exits, so every invocation is the OS's "first" render-producing reset and
stays inside the verified-safe zone regardless of how many examples get
collected in total. Slow (~6s/example, since each is a fresh SAPIEN+torch
boot) and wouldn't scale past toy sizes, but correctness matters more than
speed here.

**Not a generalization test.** D-021 pinned this env's scene layout for
good reason (G1's placement is only valid on one apartment layout), so
every example here is visually almost the same scene — the only real
variation is which object is being asked about and whether the scripted
intervention has fired. 100% accuracy on 8 easy, low-noise examples is the
minimum bar this stage needed to clear, not evidence the representation
generalizes to new objects, layouts, or lighting.

Grown to a 20-example headline result (D-026 changelog, same 100% LOO
accuracy) — the test itself stays smaller (6+6) to bound runtime.

## Second scene layout (2026-08-01)

D-027: a second calibrated apartment layout, "kitchen_sink"
(`build_config_idx=55`), added directly in response to the "not a
generalization test" caveat above — one scene is still not a real
distribution, but it's a genuine second data point instead of none.
Selected via `tidy_up_env_replicacad_humanoid.py`'s new `scene_variant`
constructor argument (`"kitchen_cabinet"` default, so every existing call
site is unaffected). Found the same way `build_config_idx=59` originally
was — but this time searched under the class's *real* two-pin
`torch.manual_seed` pattern from the start (D-021's own lesson: a naive
single-pin search gives different, wrong answers, learned once already and
nearly repeated here).

Calibration used a more precise method than the original scene's
(which found crops by eye): projected each object's known world position
through the render camera's own intrinsic/extrinsic matrices to get exact
pixel coordinates. Needed this time because `potted_meat_can` turned out
to be sitting inside a sink basin — small, low-contrast, and easy to miss
by eye at this resolution; `master_chef_can` sits in the open on a
counter, more like the original scene. `clip_feasibility.py`'s `_OBJECT_VISUAL_CONFIG`
is now keyed per scene variant; `dinov2_probe.py`'s
`collect_labeled_examples()` takes the same `scene_variant` argument.

Result: zero-shot CLIP matches oracle feasibility on "kitchen_sink" the
same way it did on "kitchen_cabinet" (`test_vision_kitchen_sink.py`).
Deliberately *not* recalibrated for this layout: reach configs, tray
position, or the goal graph — this addition is vision/rendering-only;
using "kitchen_sink" with the reach-dependent policy baselines is
untested. `test_vision_kitchen_sink.py` uses subprocess-isolated capture
(like dinov2_probe.py), not in-process rendering — test_clip_feasibility.py
already spends this process's entire D-022 render-budget (2) on the
original scene, so testing a second variant in the same process would
exceed it.

## Learned policy (2026-08-01)

`rl_policy.py` is stage 5 of the build-up order in
`docs/00-project-overview.md`: replace the scripted/oracle policies with
one that's actually learned. `feasibility_aware_policy`
(`atr.envs.tidy_up_policies`) always implemented "attempt iff feasible" as a
hard-coded rule; this checks whether an agent can discover that same rule
from trial and reward instead of being told it.

Deliberately narrow: tabular Q-learning over `(goal_id, feasible) ->
{SKIP, ATTEMPT}`, trained across 120 episodes with randomized interventions
(present or not, timing varied), using real environment rollouts — the
same `attempt_goal()` reach mechanic every other policy file uses,
unchanged. Trains in about 19 seconds on CPU, no GPU needed, since it
operates entirely on privileged state — no rendering anywhere in this
file, so D-022's confirmed upstream rendering bug never comes into play.

Result: the learned greedy policy converges to exactly "attempt iff
feasible" and matches `feasibility_aware_policy` head-to-head — same goals
achieved, zero wasted steps after `bowl_destroyed`, vs. static's 25. The
interesting part isn't the number, it's that nothing told the agent this
rule; it's recovered from reward alone.

**A real bug, found and fixed while building this:** every deterministic
baseline in this project always attempts the first goal unconditionally
(it's always feasible), so the second goal's feasibility check always
happens after a fixed elapsed time, by which point the intervention (fixed
onset step) has always already fired. A Q-learning agent explores —
including sometimes skipping the first goal — which shortens that elapsed
time and can make the second goal's feasibility check read `True`
correctly, only for the intervention to fire *during* the resulting
attempt. This produced a systematic negative bias in one Q-value (measured:
`("place_bowl", True)` converged to -0.98 instead of +1.0), not just noise
from a handful of unlucky episodes. Fixed by making a skipped action
consume the same elapsed time an attempt would have, so the environment's
timing no longer depends on which action gets explored.

**Toy-scale by construction**, same caveat as every other stage: 2 goals,
3 meaningful Q-table entries. The demonstration is that the same behavior
D-014 got by hard-coding a rule can instead be recovered by trial-and-reward
learning on real rollouts, cheaply — not a general RL result. Extending
this to a real state space (vision/representation-derived feasibility
instead of privileged-state, more goals, ordering) is future work.

## Real IK retry (2026-08-01)

D-028: retried D-024's real contact/tactile grasp confirmation, this time
with a proper tool. D-024's finite-difference IK was unreliable — the same
starting state converged to 11cm from the target on one run and 57cm on
another, no code difference. `ik_solver.py` replaces it with a real
analytic Jacobian, computed by `pinocchio` directly against G1's URDF
kinematic chain (`pin.computeFrameJacobian`), stepped via damped
least-squares — standard, numerically stable IK, not a numerical
approximation of one.

Verified before trusting it for anything, not assumed: pinocchio's
forward kinematics for `right_tcp_link`, in the URDF's own local frame,
matches `agent.right_tcp.pose.sp.p - agent.robot.pose.sp.p` (ManiSkill's
world tcp minus world base) to 5 decimal places — confirms G1's base has
zero rotation when placed via `sapien.Pose(p=...)`, and that pinocchio's
joint ordering lines up with `agent.body_joints` index-for-index (both
come from the same URDF).

Result: **fully deterministic** — identical distance across 5 repeated
runs on the same input, unlike the original attempt. Searched with
random-restart initialization (10+ restarts) across 30+ candidate base
positions (raycast floor-clearance-checked, same method as D-018), and
found neither `potted_meat_can` nor `master_chef_can` ever comes within
~13cm of the tcp — not joint-limit bound (checked directly: no arm joint
sits at its limit at convergence). Closer base positions than the
original made the residual distance *worse*, not better — moving in
forces awkward elbow/shoulder angles rather than helping. The two objects
are also ~0.6m apart, wider than the arm's functional reach envelope from
any single standing spot, so no repositioning brings *both* within range
at once.

This is now a confirmed structural limit — arm length vs. object
separation, checked from every reasonable standing position with a
solver that's actually trustworthy — not an open question a better solver
might still resolve. `teleport-on-success` remains unchanged everywhere.
`ik_solver.py` is kept as a real, tested module (not thrown away) —
`test_ik_solver.py` locks in both the kinematics-matching verification and
the unreachability finding as regression tests, and the solver itself is
reusable if this project ever needs real IK for a different object/scene
combination where the geometry might actually allow it.

## Everything combined (2026-08-02)

D-029: stage 6 of the build-up order, and the last one — combine language,
vision, and the learned policy into one real episode instead of five
demonstrations that happen to share a codebase. `end_to_end.py`, for each
goal: `parse_instruction()` (D-019/D-026) turns the instruction into a
`GoalGraph`; a real rendered frame plus `visual_object_exists()` (D-020)
judges feasibility — not a privileged-state read; a Q-table trained the
same way D-025's was (retrained here for this env's parser-generated goal
ids, since `place_potted_meat_can`/`place_master_chef_can` aren't the same
ids as the canonical env's `place_mug`/`place_bowl`) decides attempt vs.
skip from that *perceived* feasibility; `attempt_goal()` executes the
decision with real arm motion, unchanged.

Result: matches oracle exactly on a live episode. `potted_meat_can`
(perceived feasible, matches oracle) gets attempted and achieved.
`master_chef_can` (perceived infeasible after the scripted destruction,
matches oracle) gets skipped at zero cost. The same H2 result every
earlier stage already produced — the point here isn't a new number, it's
that nothing in the *live* decision path reads privileged state anymore.

Training itself still reads privileged state — disclosed, not hidden.
Training the decision *rule* ("attempt iff feasible") doesn't need real
pixels for this toy case; training against real rendered rollouts would
need hundreds of render-producing resets, which D-022's confirmed upstream
rendering bug makes impractical. So: cheap privileged-state training,
genuinely perceptual evaluation.

**Found and fixed the same bug D-025 already found once**, by not
re-applying its own fix here: skipping the first goal via exploration
shortens elapsed time before the second goal's feasibility check, which
can read `feasible=True` correctly at check-time only for the intervention
to fire mid-attempt. Caught directly, not assumed — `("place_master_chef_can",
True)` converged to a *negative* Q-value, which should never happen for a
feasible goal. Fixed the same way (`_wait()`, keeping elapsed time
consistent regardless of which action gets explored).

**Later wired in for real (D-054/D-055):** `dinov2_probe.py`'s DINOv2
probe (D-023) as an alternative to CLIP for the live perception step,
via `run_end_to_end_episode_dinov2()`. Not a clean success at first — a
real live episode surfaced a genuine robustness gap (the probe,
calibrated only on arm-at-rest captures, misjudged a destroyed object as
present once G1's own reaching arm entered frame). Closed for real
(D-055) by training on examples that reflect that same post-first-attempt
state, not by tuning the failing test. See D-054/D-055 in
`ai-notes/decisions.md` — this module remains not promotion-ready
regardless, since a closed gap in one scenario isn't a general
readiness claim.

## What this deliberately doesn't cover yet

- ~~Ordering/priority and conditional goals~~ — filled in (D-026):
  `instruction_parser.py` handles both. `Goal.condition` (conditional
  goals) reviewed and accepted as-is (D-037, kept scoped to object
  existence — see `ai-notes/review-request-task-schema.md`).
- **Preferences** (soft, non-binding wishes, as opposed to hard goals/constraints)
  — docs/04 asks for these too; no schema field exists for them, and adding
  one is a schema decision of the same size as `Goal.condition` was —
  not attempted without a driving case, same discipline as everything else here.
- ~~`depends_on` (goal ordering dependencies, distinct from `priority`) is
  still just a schema field~~ — filled in (D-037):
  `goal_dependencies_satisfied()` in `src/atr/feasibility/oracle.py`,
  exercised by `dependent_goals_example()` and wired into
  `atr.envs.tidy_up_policies`'s `feasibility_aware_policy()`. Verified live: a
  goal gets blocked by an unmet prerequisite even when its own target
  object is independently feasible. Also found while fixing this:
  `Goal.priority` is set by the parser but read by nothing — harmless
  today (execution order already matches it by construction), but worth
  knowing.
- ~~Actual goal completion~~ — filled in: `goal_achieved()` checks placement
  (object resting within the tray's footprint), used by `atr.envs.tidy_up_policies`.
  Still simplified: a successful attempt teleports the object onto the tray
  rather than re-running a full physical grasp-place sequence (see
  `atr.envs.tidy_up_policies`'s scope note) — real placement precision/collision
  between multiple placed objects isn't tested. D-024/D-028 tried to add
  real contact-based grasp *confirmation* on top of this teleport
  abstraction and found it's not achievable from G1's current position —
  see "Real IK retry" above; this remains simplified everywhere.
- ~~Held-out paraphrases/compositions~~ — filled in for the language layer
  (see above). Still not exercised for anything downstream of parsing
  (vision, policy) — those don't exist yet either.
- ~~Single scene layout for vision/representation~~ — partially filled in
  (D-027): a second calibrated layout exists now, but two scenes is still
  not a real distribution over layouts.

## How to run it

```bash
pyenv activate .maniskill
pip install -e . --no-deps
python -m pytest tests/drafts/ -v
```

The panda and humanoid variants need nothing extra. The ReplicaCAD variant
needs three additional asset downloads (~4.4GB total) before its tests will
pass instead of erroring on a missing file:

```bash
python -m mani_skill.utils.download_asset ReplicaCAD -y
python -m mani_skill.utils.download_asset ReplicaCADRearrange -y
python -m mani_skill.utils.download_asset ycb -y
```
