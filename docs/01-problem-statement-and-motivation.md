---
title: Problem Statement and Motivation
status: draft
last_updated: 2026-07-24
---

# Problem Statement and Motivation

## 1. What "task recovery" means here

A humanoid executing a task follows an implicit or explicit **plan**: a sequence of sub-goals (approach → grasp → lift → transport → place) or a continuous control policy trained to reach a goal state. "Recovery" is the process of:

1. **Detecting** that the current state has diverged from the set of states the plan/policy can handle (a *failure*), and
2. **Acting** to return to a state from which the original goal is still reachable, or **deciding** the goal is no longer reachable and aborting safely.

This is distinct from **robustness** (a policy that tolerates disturbance without ever leaving its training distribution) and from **replanning** in the classical TAMP sense (which assumes a symbolic planner and perfect state knowledge). Task recovery explicitly assumes the *primary* policy will occasionally be wrong, and treats "notice + fix" as a first-class, learned capability.

## 2. Concrete failure modes (humanoid-specific)

| # | Failure | Example | Why it's hard |
|---|---|---|---|
| 1 | Grasp failure / slip | Object slips mid-lift due to misestimated friction | Contact sensing is noisy; failure can be silent until object is already falling |
| 2 | Occlusion / perception dropout | Hand or held object blocks the head camera | Policy trained on clean vision degrades unpredictably under partial observability |
| 3 | Balance disturbance | Bump from environment, uneven contact, cable snag underfoot | High-DoF, underactuated during single support; failure cascades fast (falls happen in <1s) |
| 4 | Contact loss during manipulation | Hand slides off a handle/door edge | Task policy has no explicit "did contact hold" signal unless engineered in |
| 5 | Environment change invalidates plan | Object moved, drawer already open, obstacle appeared | Task policy conditioned on stale state estimate keeps executing a now-wrong plan |
| 6 | Actuator degradation | Joint torque saturation, motor fault, unmodeled backlash | Manifests as tracking error that looks like disturbance but has a different right response (reduce reliance on that joint, not "push harder") |
| 7 | Whole-body coordination failure | Reaching task destabilizes stance because arm motion shifts CoM further than expected | Manipulation and locomotion/balance are usually trained as separate policies; failure lives at their interface |

Humanoids make this **harder than a fixed-base arm or a wheeled robot** because: (a) failures can propagate into a fall (safety-critical, not just task-failure), (b) the action space couples balance and manipulation, and (c) recovery behaviors themselves must be dynamically stable, not just kinematically valid.

## 3. Why current approaches fall short

- **Open-loop scripted recovery** (behavior-tree fallback branches): only covers anticipated failure types; combinatorial explosion as task library grows; doesn't generalize to novel failures.
- **Pure robustness via domain randomization**: pushes the failure boundary out but doesn't eliminate it, and gives no signal *when* the boundary has been crossed — the policy just silently does something undefined.
- **End-to-end policies with no explicit failure signal**: can implicitly learn some recovery behavior if trained with enough randomized resets, but this conflates "avoid failure" and "recover from failure" into one objective, making both harder to learn and impossible to inspect/debug — bad for both performance and for explaining the system in an interview.

## 4. Research questions this project answers

1. **RQ1 (Detection):** Can a learned failure monitor (trained on proprioceptive + visual features) detect task-relevant failures earlier and with better precision/recall than fixed thresholds, and does it generalize to failure severities/types not seen in training?
2. **RQ2 (Recovery policy structure):** Does a modular "detect → select recovery skill → execute → resume" architecture outperform (a) no recovery and (b) a monolithic policy trained under heavy domain randomization, on task success rate under injected failures?
3. **RQ3 (Generalization):** How well do failure-detection and recovery behaviors generalize to unseen failure types, task variations, and simulation parameters?
4. **RQ4 (Generalization):** Does a recovery policy trained on a curriculum of failure types transfer to *held-out* failure types (e.g., trained on slip + push, tested on sensor dropout)?

## 5. Success criteria (v1)

- A working ManiSkill3 environment suite with at least 3 tasks and a failure-injection API (see [04](04-simulation-environment-maniskill.md)).
- A failure detector with measured precision/recall/latency, beating a hand-tuned threshold baseline on at least 2 failure types.
- A recovery policy that raises task success rate under injected failures by a clearly reported margin over a no-recovery baseline (target: **+30 percentage points or more**, task-dependent — treat this as a hypothesis to validate, not a promise).
- A public repo, writeup, and demo video suitable for a portfolio (see [12](12-portfolio-and-job-strategy.md)).
- **Stretch:** recovery skills that generalize across multiple simulated humanoid morphologies or task families.

## 6. Non-goals

- Solving general-purpose autonomy or long-horizon LLM-based task planning (worth a "future work" mention, not core scope).
- Achieving state-of-the-art benchmark numbers on an existing published benchmark — the contribution here is the **system + methodology + honest evaluation**, which is what portfolio reviewers and interviewers actually probe.
- Building new hardware or a new simulator — reuse ManiSkill3 and existing robot-description assets.
