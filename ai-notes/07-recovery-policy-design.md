---
title: Recovery Policy Design
status: draft
last_updated: 2026-07-24
---

# Recovery Policy Design

## 1. Recovery as a skill library (options framework)

Each recovery behavior is an **option**: `(initiation_condition, policy, termination_condition)`. This gives a clean, inspectable structure instead of one monolithic recovery network, matching the modularity principle in [03](03-system-architecture.md).

| Skill | Initiation condition | Behavior | Termination / success check |
|---|---|---|---|
| `regrasp` | grasp-slip or contact-loss detected, object still within reach | release if partially held, re-approach, re-grasp | object stably held again |
| `re-approach` | manipulation target reachable but current approach invalid (e.g., wrong angle after perturbation) | back off, re-plan approach trajectory, retry | approach pose within tolerance of a valid grasp/interaction pose |
| `step-recovery` (balance) | CoM-vs-support-polygon margin violated / high tilt rate | execute a stabilizing step or ankle/hip strategy | tilt rate and CoM margin back within nominal bounds for N consecutive steps |
| `replan-and-retry` | planning/state failure detected (environment changed) | re-perceive state, regenerate sub-goal sequence from current state | new plan validated feasible, or falls through to abort |
| `abort-to-safe-pose` | any failure detector confidence below action threshold, or recovery attempt itself times out / fails repeatedly | move to a predefined safe/neutral pose, stop | robot at safe pose, task flagged failed (not silently retried forever) |

`abort-to-safe-pose` must always be initiable and must always be able to preempt any other skill — it is the safety backstop, not just another option to be learned into disuse.

## 2. Two implementation tracks per skill (pick based on skill type)

- **Model-based / classical control** for the safety-critical, well-understood physics case: `step-recovery` is best implemented (at least for v1) via a whole-body MPC or a capture-point/ZMP-based push-recovery controller rather than pure learned RL — this is dynamically safety-critical, well-studied classically, and gives you a reliable behavior to fall back on while you iterate on the learned parts elsewhere. A learned residual/refinement on top of this classical controller is a good v2 stretch.
- **Learned (RL) policies** for skills where the "right" behavior is hard to specify analytically and failure isn't safety-critical in the same way: `regrasp`, `re-approach`, `replan-and-retry` are natural RL targets — trained per-skill in ManiSkill3 with a curriculum over the relevant failure severities.

Being explicit about *which* skills are learned vs. classical (and why) is one of the more sophisticated engineering judgment calls in this project — highlight this design choice in the writeup rather than presenting everything as "trained with RL" by default.

## 3. RL formulation for learned recovery skills

- **State**: proprioceptive state + task-relevant features (object pose relative to gripper, goal pose) + the failure signal/type from the monitor (skills can condition on *why* they were invoked).
- **Action**: same action space as the task policy (joint position/torque targets via the whole-body controller interface) — recovery skills output through the same low-level interface as the task policy so switching between them is seamless.
- **Reward shaping** (per skill, tune independently):
  - Dense shaping term toward the skill's specific sub-goal (e.g., distance to a valid grasp pose for `re-approach`).
  - Penalty for control effort / jerk to encourage smooth, efficient behavior.
  - Large terminal bonus for reaching a state where the **task policy's own termination/success check would resume normally** — the actual objective is "hand control back to the task policy successfully," not just "look locally correct."
  - Safety penalty (e.g., large negative for falling, exceeding torque limits) applied globally regardless of skill.
- **Algorithm**: start with **PPO** for stability/ease of debugging; consider **SAC** for skills needing more sample efficiency once the environment and reward are validated. Keep the same algorithm across skills initially to reduce the number of moving parts you're debugging at once.
- **Curriculum**: train each skill starting from mild-severity failure onset states, increasing severity as success rate crosses a threshold (e.g., >80% at current severity before increasing) — mirrors standard curriculum-RL practice and gives you a clean severity-vs-success curve for the writeup "for free."

## 4. The arbiter (skill selection)

- **v1 (rule-based)**: a simple mapping from `failure_signal.failure_type` (from [06](06-failure-taxonomy-and-detection.md)) to a candidate skill, filtered by each skill's `can_initiate`. Ties/ambiguity resolved by a fixed priority order (safety skills like `step-recovery` and `abort-to-safe-pose` always preempt manipulation-recovery skills).
- **v2 (learned, stretch)**: treat skill selection itself as a small learned policy (contextual bandit or shallow RL policy over the discrete skill set) trained on outcomes (did the selected skill lead to successful task resumption). Only pursue this after v1 is solid and evaluated — it's a natural "and then I improved the arbiter itself" extension for a writeup, not a v1 requirement.

## 5. Handling repeated/failed recovery attempts

Define an explicit **retry budget** per episode (e.g., at most 2 recovery attempts per failure event, at most 3 total per episode) after which the arbiter forces `abort-to-safe-pose`. Without this, a poorly-trained recovery skill can oscillate (detect failure → attempt recovery → fail again → detect failure → ...) indefinitely, a subtle bug that's easy to miss if you only look at aggregate success rate. Log retry counts explicitly as a metric ([10](10-evaluation-and-benchmarks.md)).

## 6. Interfaces recap

Recovery skills implement the `RecoverySkill` protocol from [03-system-architecture.md](03-system-architecture.md) §3 — write each skill's `can_initiate`/`step`/`is_terminated`/`succeeded` methods to that contract from the start so the arbiter never needs skill-specific special-casing.
