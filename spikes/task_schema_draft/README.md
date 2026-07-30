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

## What this deliberately doesn't cover yet

- **Language.** `GoalGraph.instruction_text` is a fixed string, not parsed
  from free text — that's explicitly Person A's territory (visual/language
  model selection, per STATUS.md), not drafted here.
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
- **Held-out paraphrases/compositions** docs/04 asks for — there's exactly
  one instruction here; templating/holdout is meaningless before the schema
  itself is agreed on.

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
