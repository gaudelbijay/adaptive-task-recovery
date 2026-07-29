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

## What this deliberately doesn't cover yet

- **Language.** `GoalGraph.instruction_text` is a fixed string, not parsed
  from free text — that's explicitly Person A's territory (visual/language
  model selection, per STATUS.md), not drafted here.
- **Priorities/dependencies as anything other than a schema field.** The
  dataclasses have `priority` and `depends_on` fields per docs/04's
  requirement, but this one example doesn't exercise them (both goals are
  equal-priority, no dependency between them) — worth a second example that
  does, once this schema shape gets buy-in.
- **Actual goal completion.** `goal_feasible()` checks whether a goal is
  still *possible*, not whether it's been *achieved* — placement success
  detection (object on tray, within some region) isn't implemented, since
  that's a manipulation-skill question layered on top of this, not a
  schema question.
- **Held-out paraphrases/compositions** docs/04 asks for — there's exactly
  one instruction here; templating/holdout is meaningless before the schema
  itself is agreed on.

## How to run it

```bash
pyenv activate .maniskill
pip install -e . --no-deps
python -m pytest tests/drafts/ -v
```
