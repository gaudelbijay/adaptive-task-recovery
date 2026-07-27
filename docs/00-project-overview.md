---
title: Project Overview
status: draft
last_updated: 2026-07-26
---

# Adaptive Task Recovery

## One-liner

A feasibility-aware vision-language reinforcement learning agent that recognizes
which instructed goals survive an irreversible world change and adapts its
strategy to maximize valid goal completion without betraying the instruction.

## Core research question

> Can a vision-language reinforcement learning agent, equipped with
> self-supervised visual representations, learn to identify which
> language-specified goals remain feasible after unforeseen and irreversible
> world changes, and adapt its task strategy to maximize goal achievement
> without violating the original intent?

Spelling in the supplied question has been normalized (`identify`, `unforeseen`,
`adapt`) without changing its meaning.

## Motivation

Long-horizon embodied agents usually assume that the world remains compatible
with their initial plan. In practice, an object may break, disappear, become
inaccessible, or be consumed; a passage may close; or an action may permanently
change another goal's feasibility. Continuing the original plan wastes effort,
while maximizing raw reward can produce a superficially successful action that
violates a constraint or changes the instruction's meaning.

ATR treats adaptation as three linked problems:

1. **Represent the changed world** from pixels using self-supervised features.
2. **Estimate goal feasibility** for each language-specified goal and constraint.
3. **Adapt strategy under intent constraints**, including justified partial completion.

## Formal view

An instruction is represented as goals `G = {g1, ..., gn}`, hard constraints
`C`, and optional priorities or preferences `P`. After an unannounced,
persistent intervention changes latent world state `z` to `z'`, the agent
observes pixels and history, estimates `Pr(feasible(gi) | o<=t, instruction)`,
and selects actions maximizing weighted valid goal achievement subject to `C`.
It must not obtain reward by silently redefining a goal.

## Scope (v1)

- Simulation-only, visually observable, object-centric tasks executed by a humanoid
- A humanoid-capable simulator, model, and library of reusable whole-body skills
- Natural-language instructions containing multiple goals and constraints
- Exogenous and action-induced irreversible changes
- Self-supervised visual pretraining or adaptation from unlabeled observations
- Held-out objects, layouts, paraphrases, and intervention types

The feasibility and intent components are designed to be embodiment-agnostic,
but v1 must include evaluation on a simulated humanoid. A simpler arm or abstract
environment may be used as a debugging testbed, not as the final evidence.

Out of scope for v1: real-robot deployment, training low-level dynamic locomotion
from scratch, unrestricted natural-language dialogue, open-web knowledge,
irreversible changes that cannot be visually or historically inferred, and
claims of general human-value alignment.

## Conceptual pipeline

```text
pixels + history --> self-supervised visual encoder --> world representation
language instruction --> goal/constraint encoder ------------------+
world representation + encoded goals --> feasibility estimator     |
                                                                   v
                                                  adaptive RL policy/planner
                                                           |
                                                intent guard/action mask
                                                           |
                                                        action
                                                           |
                                        environment + irreversible changes
```

## Primary deliverables

- A benchmark generator with reproducible, ground-truth world changes
- An oracle goal-feasibility and constraint checker
- Static, oracle, and adaptive policy baselines
- Self-supervised representation comparisons and probing results
- Multi-seed evaluation of feasibility prediction, goal achievement, intent
  violations, adaptation efficiency, and held-out-change generalization

## Document map

The numbered documents cover the problem definition, related work, architecture,
environment design, world-change taxonomy, policy design, training, evaluation,
roadmap, portfolio packaging, and experiment logging. The `ai-notes/` directory
tracks live decisions, risks, status, and work items.
