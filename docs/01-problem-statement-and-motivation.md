---
title: Problem Statement and Motivation
status: draft
last_updated: 2026-07-26
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
  only through task reward and standard supervised features.
- **H2 — explicit feasibility:** conditioning strategy selection on per-goal
  feasibility estimates outperforms a static language-conditioned policy after
  irreversible changes. **First toy-scale test (2026-07-29):** see D-014 in
  `ai-notes/decisions.md` and `spikes/task_schema_draft/policy_baselines.py`
  — a hand-authored single scenario, not evidence for the general claim, but
  the first time this hypothesis has been run rather than just stated.
- **H3 — intent guard:** explicit goal/constraint checking reduces semantic and
  constraint violations with an acceptable trade-off in achievable-goal recall.
  **First toy-scale test (2026-07-29):** see D-015 and
  `spikes/task_schema_draft/intent_guard.py` — blocks one hand-authored
  constraint violation at zero recall cost; does not yet test the harder
  recall/safety trade-off the hypothesis is actually about (see R-010).
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
