# Decisions

Lightweight architecture decision log. Stable research design is in `docs/`.

## D-009: ManiSkill3 humanoid spike — findings, not a simulator selection

- **Date:** 2026-07-28
- **Status:** Accepted
- **Decision:** Ran the Phase 0 simulator spike D-006 calls for, against
  ManiSkill3 specifically: `spikes/maniskill_humanoid_spike/` (deliberately
  outside `src/`, since D-006 says not to commit simulator-specific
  architecture yet). Confirms humanoid asset support (Unitree G1 bundled, H1
  one download away), exact deterministic seeding of a scripted event, and
  privileged-state access. Does **not** confirm object-level intervention
  support, RGB/language integration, or the skill library — and Isaac Lab
  hasn't been spiked at all yet.
- **Reason:** Needed concrete evidence before the simulator decision could be
  anything but a guess; D-006 explicitly requires this spike step.
- **Consequences:** ManiSkill3 remains a candidate, not a selection — I-003
  stays open until Isaac Lab gets an equivalent spike and the untested
  requirements (interventions, RGB, skills) are checked. Also recorded:
  no CUDA on the primary dev machine (Apple M4 Max), so SAPIEN's
  GPU-vectorized backend is unavailable there; CPU backend is fine for
  single-env dev (~450–600 steps/sec) but large-scale parallel RL training
  will need a CUDA machine regardless of which simulator is chosen. See
  `spikes/maniskill_humanoid_spike/README.md` for full results.

## D-008: Two-person ownership with shared benchmark first

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Both contributors build the benchmark and contracts first.
  Person A then leads representation/language/feasibility; Person B leads
  policy/humanoid execution. Integration and final evaluation remain shared.
- **Reason:** This balances specialization with the need to test the research
  question at the perception-policy boundary and avoids late integration.
- **Consequences:** Person A develops against recorded trajectories, Person B
  against oracle beliefs, interfaces are versioned, and roadmap phases contain
  explicit integration gates.

## D-007: Simulated humanoid is the required target embodiment

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Keep feasibility and intent reasoning embodiment-agnostic, but
  require final evaluation on a simulated humanoid using a stable skill interface.
- **Reason:** The project is intended to apply to humanoids without conflating
  high-level strategy adaptation with learning whole-body control from scratch.
- **Consequences:** Simulator selection must support humanoids; Phase 0 validates
  an asset and low-level skills; results separate skill failure from incorrect
  infeasibility; simpler embodiments may be used only as intermediate testbeds.

## D-004: Feasibility-aware vision-language RL research direction

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Study whether a vision-language RL agent using self-supervised
  visual representations can infer goal feasibility after unforeseen,
  irreversible world changes and adapt without violating the original intent.
- **Reason:** This is the project's new primary research question.
- **Consequences:** The previous humanoid failure-monitor and recovery-skill
  architecture is superseded. Environment, modules, metrics, roadmap, and
  diagram must support language goals, feasibility, and intent constraints.

## D-005: Operational definition of intent

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Represent original intent as atomic goals, dependencies,
  priorities, hard constraints, and explicit substitution/equivalence rules.
- **Reason:** “Intent” must be machine-checkable for training and evaluation.
- **Consequences:** Claims are limited to this schema and must not imply general
  human-intent understanding.

## D-006: Simulator remains undecided

- **Date:** 2026-07-26
- **Status:** Accepted
- **Decision:** Evaluate candidate object-centric visual environments before
  selecting a primary humanoid-capable simulator.
- **Reason:** ManiSkill was chosen for the old humanoid-control question; the new
  study prioritizes intervention control, language, and oracle feasibility.
- **Consequences:** Phase 0 includes a simulator spike. No simulator-specific
  architecture should be committed before it passes the selection criteria.

## D-001: Simulation-only scope

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Develop and evaluate v1 in simulation.
- **Reason:** Enables reproducible interventions and privileged oracle labels.
- **Consequences:** Claims do not extend to real robots without further evidence.

## D-002: ManiSkill as primary simulator

- **Date:** 2026-07-24
- **Status:** Superseded by D-006
- **Decision:** Originally selected ManiSkill for humanoid recovery experiments.
- **Reason:** No longer aligned with the revised question by default.
- **Consequences:** ManiSkill is now one candidate rather than a commitment.

## D-003: Separate stable docs from live tracking

- **Date:** 2026-07-24
- **Status:** Accepted
- **Decision:** Keep stable design in `docs/` and live notes in `ai-notes/`.
- **Reason:** They have different audiences and update rhythms.
- **Consequences:** Keep cross-links and status consistent.

## Template

```text
## D-NNN: Short title
- Date / Status / Decision / Reason / Consequences
```
