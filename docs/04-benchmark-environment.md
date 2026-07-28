---
title: Benchmark Environment and Task Design
status: draft
last_updated: 2026-07-26
---

# Benchmark Environment and Task Design

## Selection requirements

The simulator is not yet selected. A candidate must support a humanoid model,
RGB observations, object-centric interaction, controllable state transitions,
deterministic seeding, natural-language task generation, and privileged state
for oracle labels. It must expose or permit reusable navigation, reaching,
grasping, placing, and safe-stop skills. GPU vectorization is useful but
secondary to intervention control and ground-truth feasibility.

Candidates should be compared through a small spike rather than chosen from
reputation. ManiSkill, Isaac Lab, or another humanoid-capable platform may be the
final environment. A household or grid-world environment may provide a cheaper
logic testbed, but cannot satisfy the project's humanoid evaluation requirement.

## Task schema

Each episode contains:

- an initial world state and RGB observation stream;
- a language instruction compiled into atomic goals, priorities, dependencies,
  and hard constraints;
- a horizon and resource budget;
- zero or more hidden intervention events;
- oracle predicates for goal completion, feasibility, and constraint violation.

Language templates should support conjunction, ordering, exclusion, conditional
goals, and preferences. Hold out paraphrases and compositions rather than only
randomly splitting episodes from the same templates.

## Initial task family

A tabletop or compact household rearrangement domain is the recommended first
family. A simulated humanoid navigates, reaches, and manipulates objects using
reusable low-level skills. Example goals include placing several objects,
preserving fragile-object orientation, avoiding forbidden objects or regions,
and using limited containers or tools. Early oracle and perception tests may
disable locomotion, but final evaluation must exercise the humanoid embodiment.

## Humanoid validity requirements

- Keep the camera observations and language inputs available to the learned agent
  separate from privileged base, joint, contact, and object state.
- Measure low-level skill success separately from high-level feasibility errors.
- Do not label a goal infeasible merely because one grasp or controller rollout fails.
- Use multiple attempts or an oracle reachability model when generating labels.
- Include balance and collision safety as execution constraints, not as the main
  research target.

## Intervention API

```python
class WorldIntervention(Protocol):
    name: str
    def applicable(self, state: PrivilegedState) -> bool: ...
    def apply(self, state: PrivilegedState, severity: float) -> ChangeRecord: ...
    def persists(self, future_state: PrivilegedState) -> bool: ...
```

Every intervention records its seed, trigger, affected entities, privileged
before/after state, and oracle effect on each goal's feasibility. Interventions
must not emit a special observation marker to the agent.

## Candidate irreversible changes

- required object removed, destroyed, or permanently locked away;
- container capacity reduced or container broken;
- route or region permanently blocked;
- tool consumed, disabled, or made inaccessible;
- object state irreversibly transformed;
- agent action consumes a shared resource needed by another goal.

Include matched **reversible** and **temporary** changes. Otherwise the model may
learn that every detected change implies abandonment.

## Oracle feasibility

For small domains, compute feasibility with exhaustive search or a symbolic
planner over privileged state. For larger domains, use bounded planning and label
unknown separately rather than calling planning timeout “infeasible.” Validate
oracle labels on hand-authored cases and test monotonicity where appropriate.

## Dataset splits

- in-distribution combinations;
- held-out layouts and object appearances;
- held-out language paraphrases;
- held-out intervention types;
- held-out goal-intervention compositions;
- no-change, reversible-change, and ambiguous-evidence controls.

Prevent seed, texture, event timing, and template tokens from leaking labels.
