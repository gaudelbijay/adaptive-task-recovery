---
title: System Architecture
status: draft
last_updated: 2026-07-26
---

# System Architecture

The existing image in `media/` describes the superseded humanoid-recovery
architecture and must be redrawn before it is linked from the README.

## Design principles

- Separate perception, feasibility estimation, policy adaptation, and intent
  checking so each research hypothesis is independently testable.
- Preserve temporal history: an irreversible change can be defined by the
  transition, not a single frame.
- Represent goals individually while retaining dependencies and hard constraints.
- Keep privileged simulator state out of agent observations; use it only for
  labeling, oracle baselines, and evaluation.
- Treat uncertainty explicitly and allow safe no-op or abstention when warranted.
- Isolate high-level feasibility reasoning from humanoid morphology and low-level
  control through a stable embodiment/skill interface.

## Modules

```text
VisualEncoder(o_t, history) -> VisualState
InstructionEncoder(text) -> GoalGraph(goals, constraints, priorities)
ChangeModel(history, VisualState) -> ChangeBelief
FeasibilityModel(VisualState, ChangeBelief, GoalGraph) -> FeasibilityBeliefs
AdaptivePolicy(state, GoalGraph, beliefs) -> CandidateAction
IntentGuard(candidate, state, GoalGraph) -> Action | reject | abstain
HumanoidSkillInterface(Action, proprioception) -> whole-body commands
```

The `GoalGraph` captures goal predicates, dependencies, exclusions, priorities,
and hard constraints. Feasibility is probabilistic and per goal. The adaptive
policy may select a subgoal, choose a learned skill, or emit a primitive action.
The intent guard checks known hard constraints and disallowed substitutions.
The policy initially selects semantic skills such as navigate, reach, grasp,
place, inspect, or wait. A humanoid skill layer executes them while maintaining
balance and respecting kinematic/contact limits. Low-level skill failure is
reported back as evidence; it is not automatically equated with goal infeasibility.

## Data flow

1. Encode the instruction once and update visual state at every step.
2. Compare observation history with expected transitions to infer changes.
3. Estimate each unfinished goal's feasibility and uncertainty.
4. Select a strategy that maximizes weighted feasible-goal completion.
5. Reject or mask actions that violate explicit intent constraints.
6. Log predictions, decisions, violations, and oracle labels for analysis.

## Initial interfaces

```python
class GoalSpec(Protocol):
    goals: tuple[Goal, ...]
    constraints: tuple[Constraint, ...]

class VisualEncoder(Protocol):
    def encode(self, observations: ObservationWindow) -> VisualState: ...

class FeasibilityEstimator(Protocol):
    def predict(self, state: VisualState, spec: GoalSpec) -> FeasibilityBeliefs: ...

class AdaptivePolicy(Protocol):
    def act(self, state: VisualState, spec: GoalSpec,
            beliefs: FeasibilityBeliefs) -> Action: ...

class IntentGuard(Protocol):
    def validate(self, action: Action, state: VisualState,
                 spec: GoalSpec) -> GuardDecision: ...

class EmbodimentInterface(Protocol):
    def available_skills(self, state: VisualState) -> tuple[SkillSpec, ...]: ...
    def execute(self, skill: SkillCall) -> SkillResult: ...
    def safe_stop(self) -> None: ...
```

## Proposed repository layout

```text
src/atr/
├── envs/             # tasks, interventions, oracle feasibility labels
├── language/         # instruction schema, parsing, goal graphs
├── representations/  # self-supervised visual encoders and probes
├── feasibility/      # change and per-goal feasibility models
├── policies/         # static, adaptive, hierarchical, and oracle baselines
├── constraints/      # intent guard and violation monitors
├── control/          # humanoid skill adapters and whole-body safety interface
├── evaluation/       # metrics, splits, counterfactual tests
└── interfaces.py
configs/
scripts/
tests/
data/                 # manifests only; generated datasets ignored
docs/
ai-notes/
```

## Key ablation boundary

The primary comparison is modular explicit feasibility versus an otherwise
matched end-to-end adaptive policy. Modularization is a hypothesis, not an
assumed advantage; report if the monolithic baseline performs better.
