# Decisions

This is a lightweight architecture decision log. Add an entry when a choice
changes scope, interfaces, dependencies, evaluation, or reproducibility.

## D-001: Simulation-only scope

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Develop and evaluate the full project in simulation. Do not plan
  physical-robot experiments or sim-to-real deployment.
- **Reason:** Keeps the project feasible, reproducible, and focused on failure
  detection and recovery research.
- **Consequences:** Generalization claims must be limited to held-out simulated
  failures, tasks, parameters, seeds, and potentially robot morphologies.

## D-002: ManiSkill as the primary simulator

- **Date:** 2026-07-24
- **Status:** Provisional
- **Decision:** Use ManiSkill as the primary environment and training platform.
- **Reason:** It supports GPU-vectorized, contact-rich robot learning and custom
  robot assets.
- **Consequences:** Phase 0 must validate installation, platform compatibility,
  and humanoid-asset support before implementation depends heavily on it.

## D-003: Separate stable docs from live tracking

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Keep stable design documents in `docs/` and frequently updated
  execution notes in `ai-notes/`.
- **Reason:** Design documents and day-to-day trackers have different update
  rhythms and audiences.
- **Consequences:** Cross-links should point to `docs/` for design authority and
  `ai-notes/` for current execution state.

## Template

```text
## D-NNN: Short title

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded
- **Decision:** What was decided?
- **Reason:** Why?
- **Consequences:** What follows from it?
```
