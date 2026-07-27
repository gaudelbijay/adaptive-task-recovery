---
title: World Changes and Goal Feasibility
status: draft
last_updated: 2026-07-26
---

# World Changes and Goal Feasibility

## Definitions

A **world change** is a transition not predicted by the agent's nominal task
model. It is **irreversible** within an episode if no allowed action can restore
the relevant prior condition within the remaining horizon and constraints.

A goal is **feasible** if at least one allowed action sequence can satisfy it
from the current belief/state within remaining time and resources without
violating hard constraints. Feasibility is therefore goal-, state-,
horizon-, resource-, and constraint-dependent.

## Taxonomy

| Dimension | Values |
|---|---|
| Cause | exogenous, agent-induced, interaction-induced |
| Visibility | directly visible, inferred temporally, partially occluded |
| Persistence | temporary, reversible, irreversible |
| Goal effect | none, strategy change, one-goal loss, cascading goal loss |
| Scope | object, relation, region, tool, resource, transition dynamics |
| Evidence | certain, noisy, ambiguous |

## Prediction target

For every unfinished goal, predict `feasible`, `infeasible`, or `uncertain`, plus
a calibrated probability and optional rationale grounded in entities and
relations. A separate change head may predict affected entities and persistence;
ablate whether this intermediate supervision improves downstream behavior.

## Representation learning

Pretrain the visual encoder on unlabeled environment trajectories. Compare:

- random initialization trained only through RL;
- standard pretrained frozen visual features;
- fine-tuned pretrained features;
- self-supervised image objectives;
- temporal change-prediction objectives;
- object-centric or relation-aware objectives.

Probe object state, relations, intervention persistence, and goal reachability.
High probe accuracy is diagnostic, not the primary success metric.

## Labels and leakage controls

Privileged simulator state and the oracle planner produce labels but never enter
agent observations. Balance positive and negative examples across visual style,
event time, and instruction template. Use counterfactual pairs in which pixels
look unusual but feasibility is unchanged, and pairs with subtle feasibility
changes, to expose shortcut learning.

## Metrics

- macro F1 and AUROC for per-goal feasibility;
- expected calibration error and Brier score;
- selective risk versus coverage for abstention;
- time from intervention to correct stable estimate;
- feasible-to-infeasible and infeasible-to-feasible confusion;
- downstream regret relative to oracle-feasibility policy.

Report all metrics by intervention type, goal structure, visibility, and held-out
split. Aggregate accuracy alone can hide the harmful error of abandoning a goal
that remains feasible.
