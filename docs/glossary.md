---
title: Glossary
status: draft
last_updated: 2026-07-26
---

# Glossary

**Adaptive task recovery** — revising a task strategy after the current world no
longer supports the original plan, while preserving achievable intent.

**Goal feasibility** — existence of an allowed action sequence that achieves a
goal from the current state/belief within remaining horizon and resources,
without violating hard constraints.

**Irreversible world change** — a persistent change whose task-relevant prior
condition cannot be restored by allowed actions within the episode.

**Goal graph** — structured representation of atomic goals, dependencies,
ordering, exclusions, priorities, object identity, and hard constraints.

**Original intent** — the goals and restrictions encoded by the instruction at
episode start. In v1 it is operationalized by formal predicates and allowed
equivalences, not unrestricted human intent.

**Intent guard / shield** — runtime component that rejects or masks decisions
known to violate an explicit constraint or unauthorized substitution rule.

**Vision-language RL** — reinforcement learning conditioned jointly on visual
observations and natural-language task specifications.

**Self-supervised visual representation** — visual features learned from
structure in unlabeled images or video, such as masking, temporal prediction,
contrast, or self-distillation, rather than manual task labels.

**World-change belief** — probabilistic representation of what changed, which
entities and relations are affected, and whether the change persists.

**Selective prediction / abstention** — allowing a model to withhold a decision
when uncertainty is high and evaluating error as a function of coverage.

**Oracle feasibility** — ground-truth or planner-derived feasibility available
only for labels, evaluation, and upper-bound baselines.

**Privileged state** — simulator information unavailable to the agent but used
to generate exact predicates, interventions, and evaluation labels.

**Compositional generalization** — performance on novel combinations of known
goals, objects, relations, language forms, or world changes.

**POMDP** — partially observable Markov decision process; relevant because
pixels and history may not uniquely reveal the current world state or change.

**Constrained MDP** — decision process optimizing return subject to limits on
costs such as hard-intent violations.

**Calibration** — agreement between predicted probabilities and empirical
frequencies; essential when feasibility beliefs drive abandonment decisions.

**Adaptation regret** — valid goal value lost relative to an oracle-informed
adaptive policy under the same instruction, state, and intervention.

**Embodiment interface** — boundary between the high-level strategy policy and
the simulated humanoid's navigation, manipulation, and safety controllers.

**Skill failure versus infeasibility** — a skill failure is one unsuccessful
execution attempt; infeasibility means no allowed strategy can achieve the goal
within the remaining constraints. The former is evidence, not proof, of the latter.
