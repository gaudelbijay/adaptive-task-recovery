---
title: System Architecture
status: draft
last_updated: 2026-07-24
---

# System Architecture

> A visual version of this module breakdown, color-coded by which roadmap phase builds each part, lives at [`media/architecture-diagram.drawio`](../media/architecture-diagram.drawio) (editable in [app.diagrams.net](https://app.diagrams.net)) with a rendered preview at [`media/architecture-diagram-preview.png`](../media/architecture-diagram-preview.png). For the runtime flow and build order as a single linear narrative instead of a module-by-module breakdown, see [05-project-flow.md](05-project-flow.md).

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
│ Perception / State     │  Privileged sim state (object pose, contact) is primary for v1
│ Estimation             │  IMU + joint encoders → base orientation, joint pos/vel/torque
│                         │  Contact sensors → per-link contact flags/forces
│                         │  If/when raw vision is added: frozen pretrained backbone (e.g.
│                         │  DINOv2) — no custom SSL pretraining, no VLM. See D-004.
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

## 4. Repo layout — module packages

The `src/atr/` scaffold below exists now (empty packages + interface stubs), ahead of Phase 0 implementation, specifically so each module can be designed, built, and tested independently against the `Protocol` contracts in §3. Each module owns exactly one directory, one `README.md` describing its scope/interface/status, and one design doc in `docs/`. A module's internals are free to change as long as it keeps satisfying its `Protocol`; that's the whole point of the split in §5.

```
adaptive-task-recovery/
├── docs/                       # design docs, kept up to date as ground truth changes
├── src/atr/
│   ├── interfaces.py           # RobotInterface, FailureMonitor, RecoverySkill, Arbiter Protocols (§3)
│   ├── envs/                  # ManiSkill3 task + failure-injection environments — owns 04-simulation-environment-maniskill.md
│   ├── perception/            # state estimation; ground-truth sim state now, frozen pretrained backbone later — see D-004
│   ├── detection/             # failure monitor models + baselines — owns 06-failure-taxonomy-and-detection.md
│   ├── recovery/              # recovery skill library + arbiter — owns 07-recovery-policy-design.md
│   ├── control/                # whole-body controller / simulator action interface
│   └── configs/               # Hydra/YAML experiment configs
├── scripts/                    # train_task_policy.py, train_detector.py, train_recovery.py, eval.py
├── tests/                       # mirrors src/atr/ — tests/envs, tests/perception, tests/detection, tests/recovery, tests/control
├── docker/
├── README.md
└── STATUS.md                   # todo / current status / recent changes, kept current
```

Each module directory's `README.md` states: what it owns, what `Protocol` it implements or consumes, what it depends on from other modules (should be interfaces only, never internals), and current status. That's what makes "work on `detection/` separately from `recovery/`" actually true rather than aspirational.

## 5. Why modular over monolithic (interview-ready justification)

A single end-to-end policy trained under heavy domain randomization *might* implicitly learn to recover from disturbances, but:

- You cannot separately measure or debug "did it detect the problem" vs "did it act correctly" — one scalar success rate hides two very different failure sources.
- You cannot swap in a safety-certified balance controller for just the recovery step while keeping a learned task policy elsewhere.
- It's much easier to explain and defend in a technical interview: you can point to a specific module and a specific metric for each design decision, rather than "the network learned it somehow."

This is the single most important architectural bet in the project — document how it holds up against results honestly in [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md), including if it turns out a monolithic baseline does better on some axis.
