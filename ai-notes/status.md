# Project Status

Last updated: 2026-07-26

## Current phase

Pre-Phase 0: revised research design complete; implementation not started.

## Summary

ATR now targets feasibility-aware vision-language RL after irreversible world
changes. Stable documents have been rewritten around self-supervised visual
representations, per-goal feasibility, strategy adaptation, and explicit intent
constraints. A simulated humanoid remains the required target embodiment through
a reusable low-level skill interface; only the old physical-recovery research
question is superseded.

## Immediate focus

Choose a tractable humanoid-capable environment and asset, validate low-level
skills, formalize the goal/constraint schema, implement one intervention and
oracle-feasibility example, and redraw the architecture.

## Project health

| Area | State | Notes |
|---|---|---|
| Research question | Stable | Revised question recorded verbatim with corrected spelling |
| Scope | Draft | Simulation-only v1; simulated humanoid required |
| Architecture | Draft | Interfaces documented; no code |
| Environment | Open | Selection spike required |
| Evaluation | Draft | Primary outcomes and controls defined |
| Experiments | Not started | No results or trained models |
| Main risk | Active | Operational validity of feasibility and intent labels |

## Next milestone

Replay and score one simulated-humanoid visual-language episode containing a
persistent world change, with tested oracle labels for every goal and constraint.
