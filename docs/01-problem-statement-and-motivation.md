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
  succeed at that same scale.
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
- **H3 — intent guard:** explicit goal/constraint checking reduces semantic and
  constraint violations with an acceptable trade-off in achievable-goal recall.
  **First toy-scale test (2026-07-29):** see D-015 and
  `src/atr/constraints/intent_guard.py` (promoted from `spikes/task_schema_draft/`,
  D-037) — blocks one hand-authored
  constraint violation at zero recall cost; does not yet test the harder
  recall/safety trade-off the hypothesis is actually about (see R-010).
  Confirmed with the same result across the same four robot/scene
  combinations as H2 (D-016–D-018, D-021). The recall/safety trade-off gap
  R-010 flags is still open as of 2026-08-01 — nothing built since has
  addressed it.
- **H4 — compositional generalization:** factorized goal and change
  representations transfer better to unseen goal-change combinations than a
  monolithic policy.
- **H5 — calibration:** calibrated uncertainty and abstention outperform forced
  binary feasibility decisions when evidence is ambiguous.

## Success criteria

The project succeeds if it delivers a reproducible benchmark and demonstrates,
across multiple seeds, that the full agent improves feasible-goal completion
over a static-policy baseline while keeping hard-constraint violations below a
predeclared threshold. Feasibility accuracy alone is insufficient: estimates
must lead to better decisions. Oracle-feasibility performance defines the headroom.

## Threats to validity

Ground-truth simulator labels may make feasibility artificially easy; visual
changes may be detectable through shortcuts; templates may not reflect real
language ambiguity; and benchmark reward may encode the desired answer. Tests
must therefore include visual counterfactuals, paraphrases, held-out
interventions, and checks for representation leakage.
