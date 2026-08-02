---
title: Benchmark Environment and Task Design
status: draft
last_updated: 2026-08-01
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

**ManiSkill3 spike (2026-07-28, extended through 2026-08-01):** see
[spikes/maniskill_humanoid_spike/README.md](../spikes/maniskill_humanoid_spike/README.md)
for results scored against the requirements above. Summary: humanoid support,
deterministic seeding, privileged state, object-level interventions (removal
+ mid-episode addition of new geometry), RGB-D observations, and basic
reach/grasp manipulation all check out on non-CUDA dev hardware. One
platform gap found: `mplib`-dependent canned motion planning doesn't build
on Apple Silicon macOS (worked around via a `pinocchio`-based IK
controller — later built into a proper analytic-Jacobian solver,
`spikes/task_schema_draft/ik_solver.py`, D-028). Natural-language task
generation isn't a simulator capability either way. Navigation is also
checked (2026-07-30, D-017): ManiSkill3 exposes a real mobile robot
(`fetch`) and a real furnished scene (`ReplicaCADSetTableTrain`), but
reliable navigation needed a path planner we built ourselves
(`spikes/task_schema_draft/navigation.py`) — a naive controller gets stuck
on real walls. A confirmed, open, unfixed upstream rendering bug was also
found (D-022, `haosulab/ManiSkill#1150`) that limits how many rendered
frames can be trusted per process on macOS for real-scene environments —
guarded around, not a blocker, but a real platform cost to know about.
Isaac Lab still hasn't been spiked — I-003 in `ai-notes/issues_and_risks.md`
now leans toward "the ManiSkill3 evidence is sufficient to formally select
it" being the more defensible call, given how much validated,
ManiSkill3-specific work now exists (D-006/D-009–D-011/D-017/D-020/D-022/D-028
in `ai-notes/decisions.md`), but that's still an open decision, not made
here.

## Task schema

**Draft implementation (2026-07-29, extended through 2026-08-01):** see
[spikes/task_schema_draft/README.md](../spikes/task_schema_draft/README.md)
for a concrete, runnable, tested version of the schema below, built around
this project's own worked example from docs/01 (mug/bowl/tray/medicine/glass).
Not committed — a starting point for review (D-013 in `ai-notes/decisions.md`,
still needing that review — see `ai-notes/review-request-task-schema.md`).
Since the first draft: toy-scale tests of H2 and H3 (D-014/D-015), confirmed
embodiment-agnostic across four robot/scene combinations (D-016–D-018,
D-021), and one full pass through this project's build-up order — language
parsing (D-019, extended with ordering/priority/conditional goals in D-026,
the last a PROPOSED schema addition needing the same review), zero-shot
vision (D-020), a self-supervised representation (D-023), and a learned
policy (D-025) — see
[docs/07-adaptive-policy-design.md](07-adaptive-policy-design.md) "Policy
variants."

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

**Draft coverage:** `spikes/task_schema_draft/` implements one irreversible
case (object destroyed: `bowl_destroyed`) matched against one
reversible/temporary case (`temporary_obstacle` — a distractor object that
appears and disappears without affecting any goal), per the requirement
above. The other candidate types (container broken, route permanently
blocked, tool consumed, resource contention) are not yet drafted.

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
