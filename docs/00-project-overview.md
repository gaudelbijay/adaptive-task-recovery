---
title: Project Overview
status: draft
last_updated: 2026-07-24
---

# Adaptive Task Recovery for Humanoid Robots (ATR)

## One-liner

A learning-based framework that lets a simulated humanoid robot **notice when a task is going wrong and recover on its own** — instead of freezing, falling, or silently failing — trained and evaluated at scale in the **ManiSkill3** simulator.

## Why this project

Most robot manipulation and locomotion policies today are trained and evaluated for the *happy path*: object is where expected, ground is flat, grasp succeeds, contact holds. Real deployments never look like that. Objects slip, sensors get occluded, terrain is uneven, cables snag, external pushes happen. The dominant industry pattern for handling this is either:

1. **Stop and ask a human** (safe, but not autonomous), or
2. **Scripted recovery branches** in a behavior tree (brittle, doesn't generalize past the failure modes the engineer anticipated).

This project builds a **general, learned layer** that sits between "task policy" and "hard failure": a **failure monitor** that detects when execution has left the distribution the task policy was trained for, and a **recovery policy** that gets the robot back into a state from which the original task can resume — or gracefully aborts if it can't.

This sits squarely in the "robot learning for real-world robustness" space that humanoid robotics companies (Figure, 1X, Apptronik, Unitree, Sanctuary, Boston Dynamics) and robot-learning research groups care about deeply, which is why it's also designed as a **portfolio centerpiece** — see [12-portfolio-and-job-strategy.md](12-portfolio-and-job-strategy.md).

## Core research question

> Given a humanoid executing a long-horizon manipulation or loco-manipulation task, can we learn to (a) detect task-relevant failures from proprioceptive + visual signals faster and more generally than hand-coded thresholds, and (b) select/execute a recovery behavior that restores task feasibility, measured by recovery success rate and time-to-recovery, across failure types never seen during training?

## Scope (v1)

- **In scope:** tabletop / near-tabletop bimanual manipulation tasks and a small set of standing-balance recovery behaviors, developed and evaluated entirely in simulation.
- **Out of scope (v1):** physical-robot experiments, sim-to-real transfer, full dynamic locomotion recovery (running, jumping), multi-robot systems, and long-horizon task-and-motion planning with an LLM planner.

## High-level pipeline

```
                ┌─────────────────────────────────────────────────────────┐
                │                      HIGH-LEVEL LOOP                     │
                │                                                         │
  Sensors ──▶  Perception/State Estimation ──▶ Failure Monitor ──┐        │
 (RGB-D,       (pose, contact, joint state)     (in-dist? OOD?)  │        │
  IMU,                                                            ▼        │
  joints,                                          ┌───────────────────┐  │
  contact) ─────────────────────────────────────▶  │  Policy Arbiter   │  │
                                                     │ task | recover |  │  │
                                                     │ abort              │  │
                                                     └─────────┬─────────┘  │
                                                               │            │
                          ┌────────────────────────────────────┤            │
                          ▼                                    ▼            │
                 Task Policy (baseline)              Recovery Policy Library│
                 (trained per-task in ManiSkill)      (regrasp / re-approach│
                                                        / step-recovery /   │
                                                        replan / abort)     │
                          │                                    │            │
                          └─────────────┬──────────────────────┘            │
                                        ▼                                   │
                             Whole-Body Controller                          │
                             (ManiSkill agent API)                          │
                                        │                                   │
                                        ▼                                   │
                              Robot / Simulated Robot                       │
                └─────────────────────────────────────────────────────────┘
```

## Tech stack (initial)

| Layer | Choice | Why |
|---|---|---|
| Simulator | ManiSkill3 (SAPIEN backend) | GPU-vectorized, thousands of parallel envs on one GPU, native rigid/contact-rich manipulation support, growing humanoid asset support |
| RL algorithms | PPO, SAC (via CleanRL or Stable-Baselines3-style custom loop) | Standard, well-understood baselines; easy to reason about in interviews |
| Robot model | Unitree G1 URDF or another ManiSkill-compatible humanoid | Realistic morphology for simulated loco-manipulation and recovery |
| Config management | Hydra or plain YAML + dataclasses | Reproducible experiment configs |
| Experiment tracking | Weights & Biases (or TensorBoard if offline) | Standard in robot learning labs |
| Containerization | Docker + devcontainer | Reproducibility, portfolio polish |
| Language | Python 3.10+, PyTorch | Ecosystem fit with ManiSkill/SAPIEN |

## Assumptions made in these notes (adjust freely)

- The project is intentionally simulation-only; no physical robot access or deployment is planned.
- The **Unitree G1** is a candidate simulated morphology, not a hardware target.
- Timeline assumes **part-time, single-contributor** effort (see [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md)); compress or parallelize if you have more time/people.

## File map

| File | Purpose |
|---|---|
| [01-problem-statement-and-motivation.md](01-problem-statement-and-motivation.md) | Precise problem definition, failure taxonomy motivation, research questions |
| [02-background-and-related-work.md](02-background-and-related-work.md) | Prior art to read/cite: failure detection, recovery/safe RL, humanoid control, sim platforms |
| [03-system-architecture.md](03-system-architecture.md) | Module breakdown, data flow, interfaces |
| [04-simulation-environment-maniskill.md](04-simulation-environment-maniskill.md) | ManiSkill3 setup, robot asset import, task/environment design, failure injection API |
| [06-failure-taxonomy-and-detection.md](06-failure-taxonomy-and-detection.md) | Failure categories, detection methods, labeling strategy, metrics |
| [07-recovery-policy-design.md](07-recovery-policy-design.md) | Recovery policy library, RL formulation, hierarchical/options framing |
| [08-training-pipeline.md](08-training-pipeline.md) | End-to-end training pipeline, compute plan, logging, reproducibility |
| [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md) | Metrics, baselines, ablations, proposed open benchmark |
| [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md) | Phased plan, timeline, risk register |
| [12-portfolio-and-job-strategy.md](12-portfolio-and-job-strategy.md) | How to package this for job-hunting: repo hygiene, writeups, talking points |
| [13-experiment-log-template.md](13-experiment-log-template.md) | Running log template for tracking experiments |
| [glossary.md](glossary.md) | Terminology reference |
| [references.md](references.md) | Curated reading list by topic |
