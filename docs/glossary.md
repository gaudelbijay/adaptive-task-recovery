---
title: Glossary
status: draft
last_updated: 2026-07-24
---

# Glossary

Working definitions for this project's vocabulary. Keep entries short; link to [references.md](references.md) for deeper reading rather than expanding definitions here.

**ManiSkill3** — GPU-parallelized robot manipulation simulation benchmark/framework built on the SAPIEN physics engine; supports custom robot import via URDF/MJCF and vectorized environments for RL. See [04](04-simulation-environment-maniskill.md).

**SAPIEN** — the rigid-body physics and rendering engine underlying ManiSkill3.

**URDF / MJCF** — Unified/MuJoCo Robot Description Format: XML formats describing a robot's links, joints, and geometry, used to import a robot model into a simulator.

**DoF (Degrees of Freedom)** — number of independent joints/axes a robot can control; humanoids typically have dozens (legs, arms, torso, sometimes hands/neck).

**Whole-body control (WBC)** — control approach that computes joint commands to achieve a task-space objective (e.g., CoM position, end-effector pose) while simultaneously respecting balance and contact constraints across the whole robot, usually via quadratic-program (QP) optimization at each control step.

**MPC (Model Predictive Control)** — control method that repeatedly solves a short-horizon optimization problem using a model of the system's dynamics, applying only the first action of each solved plan before re-solving; common for bipedal balance/push-recovery.

**Capture point / ZMP (Zero Moment Point)** — classical concepts from bipedal balance control describing, respectively, where a robot's CoM would need to step to come to rest, and the point on the ground where net moment is zero; both used to design/verify push-recovery behaviors.

**PPO (Proximal Policy Optimization)** — an on-policy, gradient-based RL algorithm; a standard, stable baseline for continuous-control robot learning.

**SAC (Soft Actor-Critic)** — an off-policy, maximum-entropy RL algorithm; generally more sample-efficient than PPO for continuous control, at the cost of more implementation/tuning complexity.

**Domain randomization** — training a policy across randomized simulator parameters (mass, friction, sensor noise, etc.) so it's robust to the fact that any single simulator configuration is only an approximation of reality or of other simulator configurations.

**Sim-to-real gap** — the performance drop a policy experiences when moved from simulation to a different (typically real, but here potentially just a differently-configured) target environment, caused by unmodeled dynamics, sensing differences, and latency.

**OOD (Out-of-Distribution) detection** — detecting that current input/state differs meaningfully from the training distribution; the core mechanism behind this project's failure monitor.

**Epistemic uncertainty** — uncertainty due to lack of knowledge/data (as opposed to inherent randomness/aleatoric uncertainty); estimated here via ensemble-model disagreement to flag novel/anomalous states.

**Options framework (Hierarchical RL)** — formalism (Sutton, Precup, Singh) where a high-level policy selects among temporally-extended sub-policies ("options"), each with its own initiation set, internal policy, and termination condition; used here to structure the recovery-skill library.

**Behavior tree** — a hierarchical, node-based structure (sequence/selector/condition nodes) commonly used in game AI and robotics to encode reactive decision logic, including scripted failure-recovery branches; this project's rule-based baseline arbiter and the "scripted recovery" evaluation baseline both draw on this pattern.

**Loco-manipulation** — control problems/policies that jointly handle locomotion (walking/balance) and manipulation (reaching/grasping) rather than treating them as independent subsystems.

**Proprioception** — a robot's internal sense of its own state: joint positions/velocities/torques, base orientation/acceleration (IMU) — as opposed to exteroception (external sensing like vision).

**VLA (Vision-Language-Action) model** — a model mapping visual observations and natural-language instructions directly to robot actions; mentioned here as future-work context for a more general, language-conditioned recovery-skill selector.

**Curriculum learning (RL)** — training strategy that progressively increases task/environment difficulty (here, failure severity) as the policy demonstrates competence at the current level, rather than training on the full difficulty distribution from the start.

**Arbiter** — this project's term for the module that decides, at each control step, whether to run the task policy, a specific recovery skill, or an abort behavior, based on the failure monitor's output. See [03](03-system-architecture.md) and [07](07-recovery-policy-design.md).
