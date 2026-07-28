---
title: Portfolio and Communication Strategy
status: draft
last_updated: 2026-07-26
---

# Portfolio and Communication Strategy

The strongest artifact is a small, reproducible research result—not a broad
architecture with no evidence. Lead with the question, experimental controls,
and what the results falsified or supported.

## Demonstration story

Show paired episodes with the same instruction and initial state. In one, the
world remains unchanged. In another, a goal becomes irreversibly infeasible.
Overlay per-goal feasibility beliefs, the selected subgoal, and any guard
decision on the simulated humanoid's first-person or scene view. Contrast the
static, full, and oracle agents, and distinguish reasoning errors from low-level
humanoid skill failures.

## Repository quality bar

- one-command environment and benchmark smoke test;
- versioned configs and splits;
- clear data/model licenses;
- baseline and ablation reproduction scripts;
- tables with uncertainty and per-seed results;
- failure gallery, including intent violations;
- model card documenting observation, language, and claim boundaries.

## Interview-ready technical themes

- why feasibility differs from anomaly detection;
- why temporal self-supervision may encode persistent change;
- how privileged simulator state is isolated from learned observations;
- how intent is operationalized and where that definition is limited;
- whether modularity helped relative to a matched monolithic baseline;
- how leakage and reward hacking were tested.

Avoid framing the system as generally understanding human intent. Describe it as
constraint- and goal-faithful within an explicit benchmark schema.
