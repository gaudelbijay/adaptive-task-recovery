---
title: Background and Related Work
status: verified-core
last_updated: 2026-08-27
---

# Background and Related Work

This document defines the literature map used for the current experiments.
Exact citations and primary-source links are in [references.md](references.md).
Numerical results from different robot embodiments and task suites are not
treated as a common leaderboard.

## Closest verified systems

- **SayCan** ranks pretrained skills using language-model relevance and
  value-function affordance grounding. It is the closest reference for
  selecting feasible high-level robot skills, but does not benchmark recovery
  after hidden persistent changes that irreversibly remove goals.
- **Inner Monologue** closes the planning loop using success, scene, and human
  feedback. It motivates replanning after execution feedback, but does not
  expose ATR's per-goal feasibility, resource-contention, and explicit
  side-effect-constraint evaluation.
- **KnowNo** calibrates when an LLM planner should ask for help using conformal
  prediction. It is the closest reference for abstention under uncertainty;
  ATR instead measures autonomous action selection after world changes and the
  resulting goal/safety tradeoff.
- **Shielding** filters or corrects actions against formal safety properties.
  It motivates ATR's intent guard. Our effect-aware audit additionally shows
  why checking only a skill's named target is insufficient when its physical
  effects can violate another object's constraint.
- **PPO and ManiSkill** provide the low-level continuous-control algorithm and
  task-specific reference hyperparameters for the non-teleport manipulation
  experiments. They are execution baselines, not recovery policies.
- **PaLM-E, RT-2, and Code as Policies** broaden embodied multimodal reasoning,
  end-to-end vision-language-action control, and language-generated robot
  programs respectively. They are important representation/execution
  comparators, but their published evaluations do not isolate hidden persistent
  changes, recomputed remaining-goal feasibility, and side-effect guards.
- **ReAct** interleaves reasoning with environment actions in interactive text
  domains. It supports the closed-loop motivation but is not a physical
  manipulation or formal intent-constraint baseline.

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

| Capability | SayCan | Inner Monologue | KnowNo | Shielding | ATR benchmark |
|---|---:|---:|---:|---:|---:|
| Grounds high-level robot skills | Yes | Yes | Yes | No | Yes |
| Uses closed-loop environment feedback | Limited | Yes | Optional | Yes | Yes |
| Calibrated abstention/help | No | Human feedback | Yes | No | Conservative baseline |
| Persistent hidden change changes goal feasibility | Not isolated | Not isolated | Ambiguity focus | Safety focus | Yes |
| Reports remaining-goal efficiency after change | No | No | No | No | Yes |
| Checks explicit side-effect constraints | No | No | No | Formal action safety | Yes |
| Includes non-teleport continuous manipulation control | Physical skills | Physical skills | Physical skills | Domain-dependent | Separate PPO track |

This table is a task-definition comparison, not a claim that ATR outperforms
these systems on their published benchmarks. The quantitative comparisons in
this repository use identical ATR cases, paired seeds, and common evaluators.

## Literature-review protocol

Record the task setting, observation modality, source of language, change model,
feasibility definition, adaptation mechanism, constraint mechanism, and
generalization split for every study. This prevents grouping superficially
similar systems that answer different questions.
