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

## D-004: Perception approach — privileged state primary, frozen pretrained backbone for vision, no custom SSL or VLM

- **Date:** 2026-07-27
- **Status:** Accepted
- **Decision:** Use privileged simulator state (object pose, contact, joint state) as the
  primary signal for detection and recovery throughout v1. If/when raw vision is added,
  use a frozen pretrained visual backbone (e.g. DINOv2) rather than (a) training a custom
  self-supervised visual representation from scratch, or (b) adopting a full
  vision-language model.
- **Reason:** Training a custom SSL encoder is a separate research project in its own
  right and would divert a solo, part-time effort away from the actual contribution
  (failure detection + recovery). A full VLM is built for language-conditioned
  reasoning, which v1 explicitly excludes (see [01-problem-statement-and-motivation.md](../docs/01-problem-statement-and-motivation.md)
  §6). State observations also parallelize far better than RGB-D for RL sample
  efficiency (see [04-simulation-environment-maniskill.md](../docs/04-simulation-environment-maniskill.md) §6).
- **Consequences:** Any visual-perception work is additive and deferred, not a
  dependency of the core v1 pipeline. Frozen-backbone features must be re-evaluated
  against ground-truth state as a baseline before being relied on for detection.

## D-005: Phase the core research question — known-failure detection/recovery is v1, generalization to unseen failures is Phase 5 stretch

- **Date:** 2026-07-27
- **Status:** Accepted
- **Decision:** Split the project's research question into a v1 question (detect and
  recover from a fixed set of injected failure types, beating threshold and
  no-recovery/scripted-recovery baselines) and a stretch question (does this
  generalize to failure types/severities/tasks never seen during training). Updated
  [00-project-overview.md](../docs/00-project-overview.md) and
  [01-problem-statement-and-motivation.md](../docs/01-problem-statement-and-motivation.md)
  (RQ3/RQ4, success criteria) to reflect this explicitly.
- **Reason:** The original single-sentence research question combined detection,
  recovery, and generalization-to-unseen-failures into one claim, which overstated
  v1 scope relative to what [11-roadmap-and-milestones.md](../docs/11-roadmap-and-milestones.md)
  already treats as a later phase (Phase 5). Making the phasing explicit in the
  headline question prevents scope creep and matches R-003.
- **Consequences:** Portfolio/writeup claims about generalization must be scoped to
  Phase 5 results specifically, not implied by the v1 system working.

## Template

```text
## D-NNN: Short title

- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded
- **Decision:** What was decided?
- **Reason:** Why?
- **Consequences:** What follows from it?
```
