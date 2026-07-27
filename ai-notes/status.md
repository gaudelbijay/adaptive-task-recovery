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

The project has two contributors. Both jointly own Phase 0, the benchmark,
schemas, and evaluation design. Person A leads representation, language, and
feasibility work. Person B leads RL policy, simulator, and humanoid execution.
Both own integration and end-to-end claims.

## Immediate focus

Jointly freeze the first interface contract and benchmark slice. In parallel,
Person A runs the initial visual/language model shortlist and Person B runs the
humanoid simulator/asset/skill spike. Rejoin at the deterministic oracle-labeled
episode integration test.

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
| Staffing | Assigned | Person A: representation/feasibility; Person B: policy/humanoid; integration shared |

## Next milestone

Jointly replay and score one simulated-humanoid visual-language episode
containing a persistent world change, with tested oracle labels, a versioned
belief interface, and separate high- and low-level outcome logs.
