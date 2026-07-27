# Project Status

Living tracker for the repository. Stable research design lives in [`docs/`](docs/);
frequently updated execution notes live in [`ai-notes/`](ai-notes/).

## Current status

**Phase:** Pre-Phase 0 — research reframing complete, implementation not started.

The project now studies feasibility-aware vision-language reinforcement learning
after unforeseen, irreversible world changes, with a simulated humanoid as the
target embodiment. The previous failure-monitor/physical-recovery question has
been superseded, but humanoid manipulation and whole-body control remain part of
the execution platform. No source code or experiments exist yet; simulator,
humanoid asset, task schema, pretrained models, and compute remain to be validated.

## Todo

- [ ] Scaffold `src/atr/`, `scripts/`, `tests/`, `configs/`, and `data/`
- [ ] Select a humanoid-capable simulator with object-centric visual tasks and language goals
- [ ] Select and validate a simulated humanoid asset and reusable low-level skills
- [ ] Define a machine-checkable schema for goals, priorities, and hard constraints
- [ ] Specify the first task family and irreversible intervention set
- [ ] Build an oracle feasibility checker for benchmark labels
- [ ] Select self-supervised visual baselines (frozen and fine-tuned)
- [ ] Implement a static language-conditioned policy baseline
- [ ] Set up experiment tracking, deterministic seeds, and configuration management
- [ ] Redraw the architecture diagram for the revised system

## Recent changes

| Date | Change |
|---|---|
| 2026-07-26 | Made simulated-humanoid compatibility a required design and evaluation milestone while keeping feasibility reasoning embodiment-agnostic. |
| 2026-07-26 | Reframed the project around feasibility-aware vision-language RL, self-supervised visual representations, irreversible world changes, and intent-preserving strategy adaptation; superseded the old physical-recovery question. |
| 2026-07-24 | Added the initial simulation-only humanoid recovery research plan and project trackers. |

## Status rule

Update this file when the phase, immediate work, blockers, or project direction
changes. Keep detailed rationale in [`ai-notes/decisions.md`](ai-notes/decisions.md).
