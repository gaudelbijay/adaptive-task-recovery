---
title: System Architecture
status: draft
last_updated: 2026-07-24
---

# System Architecture

## 1. Design principles

- **Simulator modularity**: every module talks to a thin `RobotInterface` abstraction so failure monitors and recovery policies are decoupled from a specific ManiSkill environment or robot asset.
- **Modularity over end-to-end**: perception, failure detection, and recovery are separate, independently testable/debuggable components — not one opaque network. This is a deliberate tradeoff (see [02](02-background-and-related-work.md) §3) favoring explainability and incremental progress over squeezing out the last bit of performance.
- **Fail safe by default**: if the arbiter is uncertain, the default action is "abort to a safe pose," never "keep doing the task policy's raw output."

## 2. Module breakdown

```
raw sensors
   │
   ▼
┌───────────────────────┐
│ Perception / State     │  RGB-D → object pose (segmentation + pose est. or ground truth in sim)
│ Estimation             │  IMU + joint encoders → base orientation, joint pos/vel/torque
│                         │  Contact sensors → per-link contact flags/forces
└───────────┬────────────┘
            │  state vector s_t  (proprioceptive + task-relevant object/goal features)
            ▼
┌───────────────────────┐
│ Failure Monitor        │  f(s_{t-k:t}) → {nominal, failure_type, confidence}
│                         │  models: threshold baseline, ensemble-dynamics, seq2seq anomaly head
└───────────┬────────────┘
            │  failure signal
            ▼
┌───────────────────────┐
│ Policy Arbiter         │  discrete switch: TASK | RECOVER(skill_id) | ABORT
│ (rule-based v1,        │  v1: simple state machine on failure signal + confidence threshold
│  learned v2)           │  v2 (stretch): learned option-selection policy
└─────┬─────────────┬────┘
      │              │
      ▼              ▼
┌───────────┐  ┌─────────────────────────────┐
│ Task       │  │ Recovery Policy Library      │
│ Policy     │  │  - regrasp                   │
│ (per-task, │  │  - re-approach               │
│ trained    │  │  - step-recovery / balance    │
│ baseline)  │  │  - replan-and-retry           │
│            │  │  - abort-to-safe-pose         │
└─────┬──────┘  └──────────────┬──────────────┘
      │                        │
      └───────────┬────────────┘
                   ▼
      ┌─────────────────────────┐
      │ Whole-Body Controller /  │  maps desired task-space/joint targets → low-level commands
      │ Action Interface         │  ManiSkill agent.set_action()
      └─────────────┬────────────┘
                     ▼
              Robot / Sim robot
```

## 3. Interfaces (initial contract — refine once code exists)

```python
class RobotInterface(Protocol):
    def get_observation(self) -> Observation: ...
    def apply_action(self, action: Action) -> None: ...
    def get_contact_state(self) -> ContactState: ...
    def emergency_stop(self) -> None: ...

class FailureMonitor(Protocol):
    def update(self, obs_window: ObservationWindow) -> FailureSignal: ...
    # FailureSignal = {is_failure: bool, failure_type: Optional[str], confidence: float, latency_ms: float}

class RecoverySkill(Protocol):
    def can_initiate(self, obs: Observation, failure_signal: FailureSignal) -> bool: ...
    def step(self, obs: Observation) -> Action: ...
    def is_terminated(self, obs: Observation) -> bool: ...
    def succeeded(self, obs: Observation) -> bool: ...

class Arbiter(Protocol):
    def select(self, obs: Observation, failure_signal: FailureSignal) -> Literal["task", "recover", "abort"]: ...
```

Keep these as actual Python `Protocol`/ABC definitions in `src/atr/interfaces.py` once implementation starts — the important thing now is that every later design doc (failure detection, recovery policy) is written against this contract so pieces compose instead of needing a rewrite.

## 4. Proposed repo layout (for when code starts)

```
adaptive-task-recovery/
├── docs/                       # this folder — design docs, kept up to date as ground truth changes
├── src/atr/
│   ├── envs/                  # ManiSkill3 task + failure-injection environments
│   ├── perception/            # state estimation
│   ├── detection/             # failure monitor models + baselines
│   ├── recovery/              # recovery skill library + arbiter
│   ├── control/                # whole-body controller / simulator action interface
│   ├── interfaces.py
│   └── configs/               # Hydra/YAML experiment configs
├── scripts/                    # train_task_policy.py, train_detector.py, train_recovery.py, eval.py
├── tests/
├── docker/
├── README.md
├── STATUS.md                   # todo / current status / recent changes, kept current
└── docs/13-experiment-log-template.md  # linked, not duplicated
```

## 5. Why modular over monolithic (interview-ready justification)

A single end-to-end policy trained under heavy domain randomization *might* implicitly learn to recover from disturbances, but:

- You cannot separately measure or debug "did it detect the problem" vs "did it act correctly" — one scalar success rate hides two very different failure sources.
- You cannot swap in a safety-certified balance controller for just the recovery step while keeping a learned task policy elsewhere.
- It's much easier to explain and defend in a technical interview: you can point to a specific module and a specific metric for each design decision, rather than "the network learned it somehow."

This is the single most important architectural bet in the project — document how it holds up against results honestly in [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md), including if it turns out a monolithic baseline does better on some axis.
