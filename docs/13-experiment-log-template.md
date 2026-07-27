---
title: Experiment Log Template
status: active
last_updated: 2026-07-26
---

# Experiment Log Template

## Metadata

| Field | Value |
|---|---|
| Experiment ID | |
| Date / owner | |
| Commit and dirty state | |
| Phase | |
| Hypothesis | |
| Config and split version | |
| Environment/intervention version | |
| Encoder/checkpoint provenance | |
| Seeds | |
| Hardware and wall time | |

## Protocol

- Independent variable:
- Fixed controls:
- Baselines:
- Primary metric and success threshold:
- Secondary metrics:
- Leakage/counterfactual checks:

## Results

Record per-seed feasible-goal completion, intent violations, adaptation regret,
feasibility calibration, latency, nominal performance, and confidence intervals.
Link raw metrics and artifacts rather than pasting only the best run.

## Interpretation

- Did the result support the hypothesis?
- What alternative explanation remains?
- Did the agent abandon feasible goals or exploit the reward?
- What failed qualitatively?
- What single experiment should follow?
