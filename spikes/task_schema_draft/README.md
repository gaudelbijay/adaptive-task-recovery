# Task schema + intervention set — DRAFT

**What this is:** a concrete, runnable proposal for the "Shared: select the
task family and irreversible/reversible intervention set" item in
`STATUS.md` — the single biggest bottleneck right now, since the oracle
feasibility checker, dataset splits, Person A's model work, and the
versioned interfaces all depend on it. **This is a draft for review, not a
committed benchmark environment** — nothing here should be read as final
until you and your teammate have looked at it.

## The idea in one sentence

Take the project's own worked example (docs/01
"Example"): *"Put the red mug and blue bowl on the tray, keep the medicine
upright, and do not move the glass"* — and build the smallest possible
runnable version of it, so the goal-graph schema and the oracle-feasibility
logic in docs/04-benchmark-environment.md aren't just prose anymore.

## What's here

| File | What it is |
|---|---|
| [`goal_graph.py`](goal_graph.py) | `Goal`, `Constraint`, `GoalGraph` dataclasses matching docs/04's "atomic goals, priorities, dependencies, and hard constraints." `canonical_example()` builds the docs/01 instruction as data. |
| [`oracle_feasibility.py`](oracle_feasibility.py) | Pure functions: `goal_feasible()` (exists-based, never attempted-motion-based — see "Humanoid validity requirements" in docs/04) and `constraint_violated()` (position-drift / orientation checks), plus `evaluate_goal_graph()` combining both. No simulator dependency — testable in isolation. |
| [`tidy_up_env.py`](tidy_up_env.py) | A ManiSkill3 scene (5 objects: red_mug, blue_bowl, tray, medicine_bottle, glass + an idle `panda` arm) wiring the above to real privileged state. Registered as `TidyUpTaskSchemaDraft-v1`. |
| [`policy_baselines.py`](policy_baselines.py) | `static_policy()` vs `feasibility_aware_policy()` — the first runnable test of H2 (docs/01): does checking feasibility before acting beat a policy that doesn't? Also `naive_substitution_policy()`, used by the intent guard test below. |
| [`intent_guard.py`](intent_guard.py) | `validate_action()` — the first runnable test of H3 (docs/01): does rejecting an unauthorized action before execution reduce constraint violations? |
| [`tidy_up_env_humanoid.py`](tidy_up_env_humanoid.py) | The same scene and interventions, on a Unitree G1 upper body instead of the panda arm — proves `goal_graph`/`oracle_feasibility`/`intent_guard` are genuinely embodiment-agnostic, not accidentally panda-specific. Registered as `TidyUpTaskSchemaDraft-Humanoid-v1`. |
| [`policy_baselines_humanoid.py`](policy_baselines_humanoid.py) | The same `static_policy` / `feasibility_aware_policy` / `naive_substitution_policy`, adapted to the humanoid's joint-space-only control (see below). |
| [`tidy_up_env_replicacad.py`](tidy_up_env_replicacad.py) | The same goals/interventions on a **real** furnished apartment (ManiSkill3's own `ReplicaCADSetTableTrain` scene builder, real YCB objects) with a mobile Fetch robot, instead of a hand-built scene. Registered as `TidyUpTaskSchemaDraft-ReplicaCAD-v1`. |
| [`navigation.py`](navigation.py) | A grid + Dijkstra path planner, built because a naive point-and-drive controller got physically stuck on a real wall in this scene (see "ReplicaCAD embodiment" below). |
| [`policy_baselines_replicacad.py`](policy_baselines_replicacad.py) | The same three policies, navigating (not just reaching) to each goal. |
| [`tidy_up_env_replicacad_humanoid.py`](tidy_up_env_replicacad_humanoid.py) | G1 (fixed-base, no navigation) placed in the same real apartment instead of Fetch — the direct "but this is not a humanoid robot" answer. Registered as `TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1`. |
| [`policy_baselines_replicacad_humanoid.py`](policy_baselines_replicacad_humanoid.py) | The same three policies, arm-reach only (no navigation — G1 can't move its base). |
| [`language.py`](language.py) | `parse_instruction()` — turns an instruction sentence into a `GoalGraph`, instead of writing one by hand. Controlled grammar, closed object vocabulary. Wired into `tidy_up_env.py`. See "Language layer" below. |
| [`vision.py`](vision.py) | `visual_object_exists()` — judges object presence from a rendered camera frame using zero-shot CLIP, instead of reading privileged state. Calibrated against `tidy_up_env_replicacad_humanoid.py`. See "Vision layer" below. |

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
irreversible changes."* `policy_baselines.py` runs the smallest possible
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
`tests/drafts/test_policy_baselines.py`).

This is a toy-scale, existence-only version of H2, not a publishable result:
no learned feasibility estimation (the check is a direct privileged-state
query), no language, and "wasted steps" is a simplified proxy for cost, not
a full reward/regret formulation. But it's the first time any part of this
project's central research question has been demonstrated end to end,
rather than argued for in prose.

## First H3 result (2026-07-29)

H3: *"explicit goal/constraint checking reduces semantic and constraint
violations with an acceptable trade-off in achievable-goal recall."*
`naive_substitution_policy()` in `policy_baselines.py` is the "invalid
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

`language.py`'s `parse_instruction(text, known_objects)` is stage 2 of the
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

Verified three ways (`tests/drafts/test_language.py`):

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

## Vision layer (2026-07-31)

`vision.py`'s `visual_object_exists(frame, object_id)` is stage 3 of the
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
  rendered and looked at other seeds. `tests/drafts/test_vision.py` is
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
`tests/drafts/test_vision.py` keeps exactly two render-producing resets and
both were visually re-verified against saved frames directly, not just
trusted from the CLIP score.

## What this deliberately doesn't cover yet

- **Ordering/priority and conditional goals.** docs/04 asks language
  templates to support these; no existing instruction needs them yet, so
  no grammar was added for them — would be speculative without a driving
  test case.
- **Priorities/dependencies as anything other than a schema field.** The
  dataclasses have `priority` and `depends_on` fields per docs/04's
  requirement, but this one example doesn't exercise them (both goals are
  equal-priority, no dependency between them) — worth a second example that
  does, once this schema shape gets buy-in.
- ~~Actual goal completion~~ — filled in: `goal_achieved()` checks placement
  (object resting within the tray's footprint), used by `policy_baselines.py`.
  Still simplified: a successful attempt teleports the object onto the tray
  rather than re-running a full physical grasp-place sequence (see
  `policy_baselines.py`'s scope note) — real placement precision/collision
  between multiple placed objects isn't tested.
- ~~Held-out paraphrases/compositions~~ — filled in for the language layer
  (see above). Still not exercised for anything downstream of parsing
  (vision, policy) — those don't exist yet either.

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
