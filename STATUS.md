# Project Status

Living tracker for where this project actually stands. Update this file whenever status changes — it's the fast-glance source of truth; [`docs/`](docs/) holds the stable design docs and shouldn't need to change just to reflect day-to-day progress. Decisions live in [`ai-notes/decisions.md`](ai-notes/decisions.md); active risks/open questions live in [`ai-notes/issues_and_risks.md`](ai-notes/issues_and_risks.md).

## Current status

**Phase:** Pre-Phase 0 — planning complete, implementation not started.

The full design/planning documentation exists in [`docs/`](docs/) (problem statement, architecture, ManiSkill3 environment plan, failure taxonomy, recovery policy design, training pipeline, evaluation plan, roadmap), plus a full architecture + build-phases diagram in [`media/`](media/). The `src/atr/` package scaffold exists (empty modules + interface stubs, one per architectural component — see [`docs/03-system-architecture.md`](docs/03-system-architecture.md) §4) so each module can be built independently, but no module has real logic yet. Next concrete step is the rest of Phase 0 from [`docs/11-roadmap-and-milestones.md`](docs/11-roadmap-and-milestones.md): ManiSkill3 install, humanoid asset import.

## Project health

| Area | State | Notes |
|---|---|---|
| Scope | Stable | Simulation-only failure detection and recovery; v1 vs. stretch phasing documented in [`docs/00-project-overview.md`](docs/00-project-overview.md) |
| Planning | Ready | Architecture, training, and evaluation plans exist |
| Repo scaffold | Done | `src/atr/` module packages + interfaces created |
| Implementation | Not started | No environments, detectors, or policies yet |
| Testing | Not started | Add smoke tests during Phase 0 |
| Experiments | Not started | Begin logging after simulator validation |
| Blockers | None confirmed | Simulator and robot-asset compatibility remain unknown — see R-001 |

## Todo

**Now**

- [ ] Install ManiSkill3 locally and run its example tasks
- [ ] Choose a supported Python version and dependency-management approach (I-001)
- [ ] Source and import a ManiSkill-compatible humanoid URDF/MJCF (I-002)
- [ ] Pass the standing-stability smoke test ([`docs/04-simulation-environment-maniskill.md`](docs/04-simulation-environment-maniskill.md) §2)

**Next**

- [ ] Build `PushRecoveryStand` environment (first task, per [`docs/04-simulation-environment-maniskill.md`](docs/04-simulation-environment-maniskill.md) §3)
- [ ] Set up experiment tracking (W&B or TensorBoard) and config management (Hydra/YAML) before the first real training run
- [ ] Add deterministic seeding and basic episode logging

**Later**

- [ ] Train a nominal baseline policy
- [ ] Implement threshold-based failure detection
- [ ] Add learned failure detectors, then recovery policies
- [ ] Run multi-seed evaluation and ablations

## Recent changes

| Date | Change |
|---|---|
| 2026-07-27 | Added [`docs/05-project-flow.md`](docs/05-project-flow.md): single-file map of the runtime control-loop flow, build/phase order, and module ownership. Updated the architecture diagram (`media/architecture-diagram.drawio` + preview) to match the current perception decision and add an explicit "out of scope (v1)" box for self-supervised representation learning / VLM / LLM planning. |
| 2026-07-27 | Phased the core research question and success criteria (v1 = detect + recover on known injected failures; generalization to unseen failures moved to explicit Phase 5 stretch) — see [`docs/00-project-overview.md`](docs/00-project-overview.md) and [`docs/01-problem-statement-and-motivation.md`](docs/01-problem-statement-and-motivation.md). |
| 2026-07-27 | Decided perception approach: privileged sim state primary for v1; frozen pretrained visual backbone (not custom self-supervised pretraining, not a VLM) if/when raw vision is added — see D-004 in `ai-notes/decisions.md`. |
| 2026-07-27 | Scaffolded `src/atr/` as independent module packages (envs, perception, detection, recovery, control) with interface stubs and per-module READMEs, so modules can be built separately — see `docs/03-system-architecture.md` §4. |
| 2026-07-27 | Consolidated `ai-notes/status.md`, `todo.md`, and `recent_changes.md` into this file to remove duplicate tracking; `ai-notes/` now holds only `decisions.md` and `issues_and_risks.md`. |
| 2026-07-24 | Added a detailed system architecture + build-phases diagram in `media/` (`architecture-diagram.drawio`, editable in diagrams.net, plus rendered `.svg`/`.png` previews), color-coded by which roadmap phase (P0–P6) builds each component; linked from `README.md` and `docs/03-system-architecture.md`. |
| 2026-07-24 | Renamed `ai-notes/` to `docs/`; added this `STATUS.md` as the living todo/status/changelog file; updated cross-references in `README.md` and within `docs/`. |
| 2026-07-24 | Scoped the entire project to simulation-only — removed real-hardware/sim-to-real content (`05-robot-platform-unitree.md`, `09-sim-to-real-transfer.md`) and updated all cross-references accordingly. |
| 2026-07-24 | Initial `ai-notes/` documentation set written: project overview, problem statement, related work, system architecture, ManiSkill3 environment design, failure taxonomy/detection, recovery policy design, training pipeline, evaluation/benchmarks, roadmap, portfolio strategy, experiment log template, glossary, references. |

## How to update this file

- **Current status**: one short paragraph, always reflecting *today's* state — overwrite it, don't append to it.
- **Todo**: the active near-term list (roughly, "what's next" from the current roadmap phase). Check items off or delete them as completed; don't let it accumulate stale items — cross-check against [`docs/11-roadmap-and-milestones.md`](docs/11-roadmap-and-milestones.md) if it drifts.
- **Recent changes**: append one row per meaningful change (new row at the top), newest first. This is a changelog, not a diff — keep entries to one line.
