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

**Team model:** two contributors. Phase 0 and benchmark construction are shared.
Person A then leads representation/language/feasibility; Person B leads
policy/humanoid execution. Interfaces, integration, and evaluation remain shared.

## Todo

| Owner | Task |
|---|---|
| Shared | Scaffold `src/atr/`, `scripts/`, `tests/`, `configs/`, and `data/` |
| Shared | Define versioned interfaces for goal graphs, feasibility beliefs, skills, and logs |
| Shared | Select the task family and irreversible/reversible intervention set |
| Shared | Build and test the benchmark, oracle feasibility checker, and dataset splits |
| Person A | Evaluate visual and language model candidates against accuracy, calibration, latency, memory, licensing, and downstream utility |
| Person A | Select self-supervised visual baselines and implement goal/change/feasibility models |
| Person B | Select a humanoid-capable simulator and validate the humanoid asset and low-level skills — ManiSkill3 spiked 2026-07-28 (`spikes/maniskill_humanoid_spike/`, D-009/D-010): humanoid, seeding, privileged state, AND object-level interventions all confirmed; Isaac Lab spike and RGB/skill-library testing still open |
| Person B | Implement static, oracle-feasibility, and adaptive policy baselines plus the intent guard |
| Shared | Set up experiment tracking, deterministic seeds, integration tests, and evaluation harness |
| Shared | Redraw the architecture diagram with ownership and module boundaries |

## Recent changes

| Date | Change |
|---|---|
| 2026-07-28 | Extended the spike to test object-level interventions (`object_intervention_spike.py`, D-010) — the requirement that actually gates the simulator decision, not standing balance. Confirmed on ManiSkill3: object removal from the live physics scene, and mid-episode addition of new geometry, both deterministic given a seed. Found and documented a real gotcha: the `Actor` wrapper goes stale after removal. I-003 downgraded to Medium severity. |
| 2026-07-28 | Ran the ManiSkill3 humanoid simulator spike D-006 calls for: `spikes/maniskill_humanoid_spike/` (Unitree G1 + H1, deterministic scripted push, CPU-backend sim, pytest suite, recorded videos). Findings in `spikes/maniskill_humanoid_spike/README.md` and D-009 — humanoid/seeding/privileged-state confirmed, object-level interventions and Isaac Lab comparison still open. Also resolved the Python/dependency setup in practice: pyenv virtualenv `.maniskill` (Python 3.12.12), pinned in `requirements-maniskill.lock.txt`. |
| 2026-07-26 | Adopted a two-person ownership model: shared benchmark first, then Person A leads representations/feasibility and Person B leads policy/humanoid execution, with shared integration. |
| 2026-07-26 | Made simulated-humanoid compatibility a required design and evaluation milestone while keeping feasibility reasoning embodiment-agnostic. |
| 2026-07-26 | Reframed the project around feasibility-aware vision-language RL, self-supervised visual representations, irreversible world changes, and intent-preserving strategy adaptation; superseded the old physical-recovery question. |
| 2026-07-24 | Added the initial simulation-only humanoid recovery research plan and project trackers. |

## Status rule

Update this file when the phase, immediate work, blockers, or project direction
changes. Keep detailed rationale in [`ai-notes/decisions.md`](ai-notes/decisions.md).
