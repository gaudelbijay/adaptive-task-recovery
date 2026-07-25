# Project Status

Living tracker for where this project actually stands. Update this file whenever status changes — it's the fast-glance source of truth; [`docs/`](docs/) holds the stable design docs and shouldn't need to change just to reflect day-to-day progress.

## Current status

**Phase:** Pre-Phase 0 — planning complete, implementation not started.

The full design/planning documentation exists in [`docs/`](docs/) (problem statement, architecture, ManiSkill3 environment plan, failure taxonomy, recovery policy design, training pipeline, evaluation plan, roadmap), plus a full architecture + build-phases diagram in [`media/`](media/). No code has been written yet. Next concrete step is Phase 0 from [`docs/11-roadmap-and-milestones.md`](docs/11-roadmap-and-milestones.md): repo scaffold, ManiSkill3 install, humanoid asset import.

## Todo

- [ ] Set up repo scaffold (`src/atr/`, `scripts/`, `tests/`, `docker/`) per [`docs/03-system-architecture.md`](docs/03-system-architecture.md) §4
- [ ] Install ManiSkill3 locally and run its example tasks
- [ ] Source and import a ManiSkill-compatible humanoid URDF/MJCF
- [ ] Pass the standing-stability smoke test ([`docs/04-simulation-environment-maniskill.md`](docs/04-simulation-environment-maniskill.md) §2)
- [ ] Build `PushRecoveryStand` environment (first task, per [`docs/04-simulation-environment-maniskill.md`](docs/04-simulation-environment-maniskill.md) §3)
- [ ] Set up experiment tracking (W&B or TensorBoard) and config management (Hydra/YAML) before the first real training run

## Recent changes

| Date | Change |
|---|---|
| 2026-07-24 | Added a detailed system architecture + build-phases diagram in `media/` (`architecture-diagram.drawio`, editable in diagrams.net, plus rendered `.svg`/`.png` previews), color-coded by which roadmap phase (P0–P6) builds each component; linked from `README.md` and `docs/03-system-architecture.md`. |
| 2026-07-24 | Renamed `ai-notes/` to `docs/`; added this `STATUS.md` as the living todo/status/changelog file; updated cross-references in `README.md` and within `docs/`. |
| 2026-07-24 | Scoped the entire project to simulation-only — removed real-hardware/sim-to-real content (`05-robot-platform-unitree.md`, `09-sim-to-real-transfer.md`) and updated all cross-references accordingly. |
| 2026-07-24 | Initial `ai-notes/` documentation set written: project overview, problem statement, related work, system architecture, ManiSkill3 environment design, failure taxonomy/detection, recovery policy design, training pipeline, evaluation/benchmarks, roadmap, portfolio strategy, experiment log template, glossary, references. |

## How to update this file

- **Current status**: one short paragraph, always reflecting *today's* state — overwrite it, don't append to it.
- **Todo**: the active near-term list (roughly, "what's next" from the current roadmap phase). Check items off or delete them as completed; don't let it accumulate stale items — cross-check against [`docs/11-roadmap-and-milestones.md`](docs/11-roadmap-and-milestones.md) if it drifts.
- **Recent changes**: append one row per meaningful change (new row at the top), newest first. This is a changelog, not a diff — keep entries to one line.
