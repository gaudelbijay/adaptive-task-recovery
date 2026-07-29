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
