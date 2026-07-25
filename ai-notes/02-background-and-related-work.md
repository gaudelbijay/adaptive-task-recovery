---
title: Background and Related Work
status: draft
last_updated: 2026-07-24
---

# Background and Related Work

This is a working reading list + concept map, organized by sub-area. Treat every claim below as "concept to verify by reading the primary source," not a citation you can quote from memory in an interview — read the actual papers before relying on specifics. See [references.md](references.md) for the flat reading list.

## 1. Failure / anomaly detection in robotics

- **Threshold/rule-based**: contact-force outliers, joint-tracking-error thresholds, IMU tilt-angle limits. Simple, interpretable, zero training cost — the baseline this project must beat.
- **Learned dynamics-model disagreement**: train an ensemble of forward dynamics models; large disagreement (epistemic uncertainty) on the current state/action flags an out-of-distribution situation. Common in model-based RL safety literature.
- **Reconstruction-based OOD**: train a VAE/autoencoder on nominal proprioceptive+visual trajectories; high reconstruction error at test time signals anomaly. Cheap, unsupervised, but can be insensitive to subtle task-relevant (vs. merely statistically unusual) failures.
- **Sequence models for anomaly scoring**: LSTM/Transformer trained to predict next-state; prediction error or learned anomaly head over a window of proprioceptive history. Good fit for this project since failures unfold over a short time window, not a single frame.
- **Contact/force-torque sensing literature**: classic robotics work on using wrist F/T sensors and tactile sensing to detect slip and misalignment during manipulation — relevant baseline signal source even without deep learning.

## 2. Recovery, safe RL, and "what to do after detecting a problem"

- **Recovery RL** (Thananjeyan et al.): learns a *risk-aware recovery policy* alongside a task policy — when a learned safety critic predicts the task policy is about to enter a constraint-violating region, control is handed to the recovery policy. Directly relevant structural template for this project (adapt the objective from "avoid unsafe states" to "restore task feasibility").
- **Safe RL / control barrier functions / Lyapunov-based safety layers**: a broader literature on wrapping a learned policy with a safety filter. Useful for the balance-recovery sub-problem specifically (falls are a hard safety constraint, not just a task-failure).
- **Options / hierarchical RL framework**: treat each recovery behavior (regrasp, step-recovery, re-approach) as an *option* with its own initiation set and termination condition; the failure monitor + arbiter effectively learns/implements the option-selection policy. This gives a clean, explainable architecture (good for interviews).
- **Reset-free / autonomous RL**: literature on training policies that don't rely on a human resetting the environment after every failure — relevant to how you'd eventually train recovery behaviors with less simulator hand-holding, and directly relevant if you ever try to fine-tune on a real robot.
- **Behavior trees with learned conditions**: classical robotics fallback structure; worth reading as the "why not just do this" foil in your writeup — behavior trees are still the industry-standard baseline you're arguing against/alongside.

## 3. Humanoid whole-body control and locomotion

- **Whole-body control (WBC) / QP-based controllers**: standard approach for translating a desired task-space action (e.g., "shift CoM," "step here") into joint torques while respecting balance and contact constraints. Likely needed as the low-level layer under any RL "step-recovery" skill so it stays dynamically feasible.
- **Model Predictive Control (MPC) for bipedal balance**: short-horizon optimization for push-recovery (capture point / ZMP-based methods are the classical starting point). Read at least one capture-point / ZMP push-recovery paper before implementing the balance-recovery skill.
- **Learned bipedal locomotion (sim-to-real)**: recent work training walking/loco-manipulation policies end-to-end in GPU-parallel sim (Isaac Gym/Lab, MuJoCo, ManiSkill-adjacent) with heavy domain randomization, then transferring to real bipedal/humanoid hardware. This is the closest "sibling" literature — your project specifically targets the *failure/recovery* slice of this pipeline rather than nominal locomotion.
- **Loco-manipulation**: work that jointly controls locomotion and manipulation (whole-body policies rather than separate walking + arm controllers) — relevant to failure mode #7 in [01](01-problem-statement-and-motivation.md).

## 4. Vision-language(-action) models and high-level replanning (context, not core scope)

- **SayCan (Ahn et al.)**: grounds an LLM's affordance reasoning in a learned value function to select feasible robot skills — relevant conceptually to "what do we do after failure" at the symbolic level, even though this project's recovery layer is lower-level.
- **RT-2 / vision-language-action (VLA) models**: end-to-end models mapping vision+instruction to actions; mention in "future work" as a possible way to make the recovery *skill selector* more general (natural-language-conditioned recovery) rather than a fixed discrete set.
- **Eureka-style LLM reward design**: using an LLM to help author/iterate RL reward functions; potentially useful as an engineering accelerant for shaping recovery-policy rewards, not a core research claim.

## 5. Simulation platforms — why ManiSkill3

| Platform | Physics backend | Parallelism | Manipulation focus | Humanoid/legged support | Notes |
|---|---|---|---|---|---|
| **ManiSkill3** | SAPIEN (PhysX) | GPU-vectorized, thousands of envs/GPU | Strong (its origin) | Growing — supports arbitrary URDF/MJCF robots including bipeds/humanoids via its agent API | Chosen for this project: open-source, active development, good rendering, GPU throughput needed for RL sample efficiency |
| Isaac Lab / Isaac Gym | PhysX (Nvidia) | GPU-vectorized | Good | Strong, widely used for legged/humanoid RL in industry | Heavier install footprint, Nvidia-GPU-locked; worth a comparison paragraph in your writeup even if not primary |
| MuJoCo (+ MJX) | MuJoCo | MJX offers JAX-based vectorization | Good, very accurate contact | Good (many humanoid RL papers use MuJoCo) | Excellent for controller/MPC prototyping; smaller asset ecosystem for rich manipulation scenes than ManiSkill |
| Genesis | Custom | GPU-vectorized | Growing | Growing | Newer, worth a "future alternative" mention |

**Decision:** ManiSkill3 as primary simulator, because it gives GPU-parallel RL throughput *and* rich manipulation-scene tooling in one framework, minimizing the number of systems to integrate for a solo project. Document any limitations you actually hit (humanoid asset stability, contact tuning) in [04](04-simulation-environment-maniskill.md) as you go — this is exactly the kind of "I hit X, here's how I diagnosed and worked around it" content that makes a project writeup credible.

## 6. Reading priority order

1. One classical push-recovery/balance paper (ZMP or capture-point based).
2. Recovery RL (Thananjeyan et al.) — architecture template.
3. One GPU-parallel sim-to-real humanoid/legged locomotion paper — methodology template for domain randomization and reward shaping.
4. Options framework primer (Sutton, Precup, Singh) — vocabulary for the recovery-skill-library design.
5. SayCan — for the "future work: language-conditioned recovery" paragraph.

Update [references.md](references.md) with full citations as you actually read these — don't front-load fabricated bibliographic detail.
