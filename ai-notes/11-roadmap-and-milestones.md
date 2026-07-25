---
title: Roadmap and Milestones
status: draft
last_updated: 2026-07-24
---

# Roadmap and Milestones

Assumes **part-time, single-contributor** effort. Treat durations as planning estimates to revise once you have real velocity data from Phase 0 — the very first thing you'll learn is whether these estimates are realistic for your actual available hours/week.

## Phase 0 — Setup (~2 weeks)

- Repo scaffold per [03-system-architecture.md](03-system-architecture.md) §4, Docker/devcontainer, CI skeleton (lint + a smoke test).
- ManiSkill3 installed and running its example tasks locally.
- Unitree humanoid URDF imported and standing-stability smoke test passing ([04](04-simulation-environment-maniskill.md) §2).
- **Exit criteria**: can run a random-action rollout of the imported humanoid in ManiSkill3 and record a video.

## Phase 1 — Baseline task policies (~3–4 weeks)

- Implement `PushRecoveryStand` first, then `PickPlaceRecovery`, then `DoorOpenRecovery`/`CarryWalkRecovery` as time allows.
- Train/tune Stage-0 baseline policies per task per [08-training-pipeline.md](08-training-pipeline.md).
- **Exit criteria**: all built tasks have a baseline policy at >90% nominal success rate (or documented reason a specific task is harder), logged with seeds/configs.

## Phase 2 — Failure injection and taxonomy validation (~2 weeks)

- Implement the failure-injection API ([04](04-simulation-environment-maniskill.md) §4) for at least external_force, friction_drop, object_perturbation, sensor_dropout.
- Sweep severities to find "interesting" ranges per task per [08](08-training-pipeline.md) Stage 1.
- **Exit criteria**: documented severity curves (success rate vs. severity for the *unmodified* Stage-0 policy) per failure type per task.

## Phase 3 — Failure detector (~3 weeks)

- Threshold baseline → ensemble-dynamics → sequence model, per [06-failure-taxonomy-and-detection.md](06-failure-taxonomy-and-detection.md) §5.
- **Exit criteria**: precision/recall/latency table beating threshold baseline on ≥2 failure types, committed with reproducible eval script.

## Phase 4 — Recovery policy training (~4–6 weeks)

- Classical `step-recovery` controller implemented and validated.
- RL-trained `regrasp`, `re-approach`, `replan-and-retry` with curricula, per [07-recovery-policy-design.md](07-recovery-policy-design.md).
- Rule-based arbiter wiring the full pipeline together.
- Full baseline/ablation suite from [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md) run and reported.
- **Exit criteria**: this is the core deliverable — end-to-end system beating no-recovery and scripted-recovery baselines, with honest numbers, is enough on its own for a strong portfolio piece even if nothing past this point happens.

## Phase 5 — Sim-to-real prep and hardware bring-up (~3–4 weeks, contingent on hardware access)

- Safety protocol dry-runs, system identification, domain-randomization widening, per [09-sim-to-real-transfer.md](09-sim-to-real-transfer.md).
- **Exit criteria**: shadow-mode validated, Stage A (seated manipulation-recovery) hardware sessions logged.

## Phase 6 — Real robot experiments (~4 weeks, stretch)

- Staged deployment through as many of Stages A–D as safety/access allow.
- **Exit criteria**: at minimum, Stage A results reported honestly; anything further is bonus.

## Phase 7 — Writeup, demo, polish (~2–3 weeks)

- Public repo cleanup, README with GIFs/videos, blog-style writeup(s), demo video, optional workshop-paper draft.
- Per [12-portfolio-and-job-strategy.md](12-portfolio-and-job-strategy.md).

## Total estimate

~5–6 months part-time through Phase 4 (the sim-only, fully self-contained deliverable); +2–3 months if pursuing hardware phases. **Ship Phase 4's writeup even if hardware phases stall or never happen** — don't let a stretch goal block releasing a complete, valuable simulation result.

## Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Humanoid asset import in ManiSkill is unstable/fiddly | Delays Phase 0–1 | Budget extra time explicitly (this is a known friction point, see [04](04-simulation-environment-maniskill.md) §2); fall back to a simpler/better-supported robot asset temporarily to validate the *pipeline* while asset issues are debugged in parallel |
| No real hardware access materializes | Phases 5–6 don't happen | Explicitly scoped as stretch from day one ([05](05-robot-platform-unitree.md) §4); project is complete without them |
| RL sample inefficiency / reward shaping struggles | Delays Phase 4 | Start with the simplest env (`PushRecoveryStand`) and classical control where possible ([07](07-recovery-policy-design.md) §2); don't over-invest in the hardest RL skill first |
| Sim2real gap larger than expected | Weak/no hardware results | Frame stretch-goal success criteria realistically from the start ([09](09-sim-to-real-transfer.md) §6); simulation results stand alone |
| Scope creep (adding tasks/skills indefinitely) | Timeline slips, nothing ships | Hold to the v1 scope in [01-problem-statement-and-motivation.md](01-problem-statement-and-motivation.md) §6; log new ideas as "future work," don't implement mid-stream |
| Time/motivation drop-off on a long solo project | Project stalls | Ship incrementally — each phase's exit criteria is independently postable/shareable (blog post per phase), so partial progress still produces visible portfolio artifacts |
