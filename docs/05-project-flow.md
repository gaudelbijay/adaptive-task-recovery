---
title: Project Flow
status: draft
last_updated: 2026-07-27
---

# Project Flow

This is the single file to read to understand **how the system runs, step by
step, and in what order it gets built**. Every other doc goes deeper on one
piece of this; this one is the map. See [`media/architecture-diagram-preview.png`](../media/architecture-diagram-preview.png)
for the same thing as a diagram.

## 1. Runtime flow — one control-loop timestep

This is what happens, in order, every simulation step once the system is
wired end-to-end (Phase 4+). Module names in brackets are the `src/atr/`
package that owns that step.

1. **Sensors produce a raw observation** — RGB-D, IMU, joint encoders, contact
   sensors. `[atr.envs]`
2. **Perception turns it into a state vector `s_t`** — privileged sim state
   (object pose, contact, joint state) is the v1 signal; a frozen pretrained
   visual backbone is an optional later addition. No custom-trained visual
   representation, no VLM — see §5. `[atr.perception]`
3. **The failure monitor scores the last `k` steps of state** — threshold
   baseline first, then ensemble-dynamics, then a sequence model (build order
   fixed, each must beat the previous) — producing a `FailureSignal{is_failure,
   failure_type, confidence, latency_ms}`. `[atr.detection]`
4. **The arbiter picks one of `TASK`, `RECOVER(skill_id)`, `ABORT`** from the
   failure signal + state. v1 is a rule-based state machine; a learned
   selector is a v2 stretch. Safety (abort-to-safe-pose) always preempts.
   `[atr.recovery]`
5. **Either the task policy or a recovery skill produces a target action.**
   Task policy: one baseline PPO/SAC policy per task, no failure-awareness.
   Recovery skill: one of `regrasp`, `re-approach`, `step-recovery`,
   `replan-and-retry`, `abort-to-safe-pose`, each with its own
   `can_initiate` / `step` / `is_terminated` / `succeeded`. `[atr.recovery]`
   (task policy itself is trained per-task, not owned by any one module — see
   [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md) Phase 1).
6. **The whole-body controller maps that target to joint commands** and calls
   `agent.set_action()` — the same interface for the task policy and every
   recovery skill. `[atr.control]`
7. **The simulator advances physics**, optionally with a failure injected this
   step (`external_force`, `friction_drop`, `object_perturbation`,
   `sensor_dropout`, `actuator_fault`, `contact_loss` — deterministic, seeded,
   severity-parameterized). `[atr.envs]`
8. **Loop back to step 1.** Ground-truth failure labels and arbiter decisions
   are logged to the evaluation harness throughout, for eval only — they
   never feed back into the control loop itself.

The `Protocol` contracts each module implements (`RobotInterface`,
`FailureMonitor`, `RecoverySkill`, `Arbiter`) live in
[`src/atr/interfaces.py`](../src/atr/interfaces.py); that's what lets steps
2–6 be built and tested independently of each other.

## 2. Build flow — the order modules actually get built in

Runtime flow (§1) is the *shipped* order; build order is different, because
you can't test a module against a real upstream module until that upstream
module exists. Full detail and exit criteria: [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md).

| Phase | Builds | Depends on | Exit criteria |
|---|---|---|---|
| **P0 — Setup** | Repo scaffold, ManiSkill3 install, humanoid import | — | Random-action rollout of the imported humanoid runs and records video |
| **P1 — Baseline policies** | `atr.envs` tasks + per-task `atr.control`/task-policy baseline | P0 | All built tasks ≥90% nominal success rate |
| **P2 — Failure injection** | `FailureInjector` in `atr.envs` | P1 (need something that can fail) | Documented severity-vs-success curves per failure type |
| **P3 — Failure detector** | `atr.detection` (threshold → ensemble → sequence model) | P2 (need labeled failures) | Beats threshold baseline on ≥2 failure types |
| **P4 — Recovery policy (core deliverable)** | `atr.recovery` (skills + arbiter), wired end-to-end | P3 (need a failure signal to arbitrate on) | Beats no-recovery + scripted-recovery baselines |
| **P5 — Generalization & benchmark** | Held-out eval, benchmark packaging | P4 | Held-out failure types/severities measured; benchmark runs from clean setup |
| **P6 — Writeup & polish** | Demo video, writeup, repo cleanup | P5 | Portfolio-ready public repo |

P4's exit criteria is the one deliverable the project needs to succeed even
if nothing past it happens — see [00-project-overview.md](00-project-overview.md)
and [D-005](../ai-notes/decisions.md).

## 3. Module ownership (who builds what, independently)

| Module | Runtime role (§1 step) | Design doc | Depends on (interfaces only) |
|---|---|---|---|
| [`atr.envs`](../src/atr/envs/) | 1, 7 | [04-simulation-environment-maniskill.md](04-simulation-environment-maniskill.md) | nothing else in `src/atr/` |
| [`atr.perception`](../src/atr/perception/) | 2 | [03-system-architecture.md](03-system-architecture.md) §2 | `atr.envs` observations |
| [`atr.detection`](../src/atr/detection/) | 3 | [06-failure-taxonomy-and-detection.md](06-failure-taxonomy-and-detection.md) | `atr.perception` state vector |
| [`atr.recovery`](../src/atr/recovery/) | 4, 5 | [07-recovery-policy-design.md](07-recovery-policy-design.md) | `atr.detection` failure signal |
| [`atr.control`](../src/atr/control/) | 6 | [03-system-architecture.md](03-system-architecture.md) §2 | `atr.recovery` / task-policy actions |

## 4. Data flow summary

```
raw sensors -> s_t (perception) -> FailureSignal (detection)
   -> {TASK | RECOVER(skill) | ABORT} (arbiter, in recovery)
   -> Action (task policy or recovery skill)
   -> joint commands (control) -> simulator -> raw sensors [next step]
```

Failure injection and the evaluation harness sit outside this loop: injection
perturbs the simulator directly; the harness only reads ground-truth labels
and arbiter decisions for offline metrics, per [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md).

## 5. What's explicitly not in this flow

No step above trains a custom self-supervised visual representation, calls a
vision-language model, or does LLM-based planning — that's a deliberate v1
scope decision, not an omission. See [D-004 / D-005](../ai-notes/decisions.md)
and the vision-language(-action) models background in
[02-background-and-related-work.md](02-background-and-related-work.md) §4 for
why, and what it would take to add later.
