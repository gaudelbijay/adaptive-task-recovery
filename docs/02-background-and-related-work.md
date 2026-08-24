---
title: Background and Related Work
status: draft
last_updated: 2026-07-26
---

# Background and Related Work

This document defines the literature map to verify before implementation. Exact
citations belong in [references.md](references.md); claims should not be made
from remembered titles alone.

## Vision-language embodied agents

Review language-conditioned control, vision-language-action models, embodied
instruction following, and task-and-motion planning with language. The relevant
gap is not instruction following in a static world, but recomputing the feasible
goal set after an unannounced persistent change.

## Self-supervised visual representation learning

Compare contrastive, masked-image-modeling, self-distillation, and temporal or
object-centric objectives. The useful representation must encode object state,
relations, affordances, and change—not merely image similarity. Frozen,
fine-tuned, and task-reward-only encoders are necessary baselines.

## Goal-conditioned and hierarchical RL

Goal-conditioned RL, successor features, hierarchical policies, and constrained
MDPs provide tools for multi-goal optimization and policy reuse. ATR differs by
making language-defined goal feasibility nonstationary and partially observed.

## Humanoid loco-manipulation and skill interfaces

Review reusable humanoid navigation, reaching, grasping, whole-body control, and
vision-language-action skill interfaces. These are enabling components rather
than the primary novelty. The study must distinguish controller reachability
from goal infeasibility and document the embodiment assumptions behind its oracle.

## Continual adaptation and nonstationary MDPs

Nonstationary and hidden-parameter MDP work addresses changing dynamics or task
conditions. ATR focuses on discrete, persistent world-state changes that alter
the reachable goal set, often within a single episode and without online weight
updates.

## Affordance and feasibility prediction

Affordance learning, reachability/value estimation, and precondition-effect
models motivate the feasibility head. “Feasible” must be defined relative to the
current state, remaining horizon/resources, available actions, and constraints.

## Constrained decision-making and intent preservation

Constrained RL, shielding, runtime verification, reward machines, and temporal
logic offer operational ways to prevent hard violations. The project should use
machine-checkable constraints while testing whether language encodings preserve
their link to those constraints.

## Closest-baseline matrix

| Capability | Static VLM/RL | Replanner with oracle state | ATR target |
|---|---:|---:|---:|
| Language-conditioned goals | Yes | Yes | Yes |
| Learns visual representations without labels | Optional | No | Yes |
| Detects persistent world changes | Incidental | Oracle | Learned |
| Predicts per-goal feasibility | No | Oracle | Learned |
| Adapts within the episode | Limited | Yes | Yes |
| Enforces explicit intent constraints | Usually no | Rule-based | Guard + policy |

## Literature-review protocol

Record the task setting, observation modality, source of language, change model,
feasibility definition, adaptation mechanism, constraint mechanism, and
generalization split for every study. This prevents grouping superficially
similar systems that answer different questions.
