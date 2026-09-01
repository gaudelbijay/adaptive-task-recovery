---
title: Roadmap and Milestones
status: draft
last_updated: 2026-08-01
---

# Roadmap and Milestones

## Scope areas

- **Shared:** Phase 0, benchmark/oracle construction, schemas, interface contracts,
  dataset splits, integration, final evaluation, and claims.
- **Representation:** visual/language model selection, self-supervised representation,
  goal graphs, world-change and per-goal feasibility models, and calibration.
- **Policy:** simulator and humanoid integration, reusable skills, static and
  adaptive RL policies, intent guard, and policy baselines.

Representation develops against recorded trajectories. Policy develops against
oracle feasibility beliefs. Each phase has an integration gate so learned
beliefs replace the oracle incrementally rather than at the end.

## Phase 0 — foundations

- **Shared:** Scaffold the repository, pin dependencies, define schemas, and
  agree on the `AgentBelief` and humanoid skill contracts.
- **Representation:** Compare visual/language model candidates with a small offline probe.
- **Policy:** Compare humanoid-capable environments, import an asset, and
  validate reusable low-level skill interfaces.
- **Shared:** Integrate both spikes into a minimal visual-language task.

**Exit:** one deterministic humanoid episode can be replayed, rendered, and
scored, with high- and low-level outcomes logged separately.

**Status as of 2026-08-01:** substantially demonstrated in `spikes/`, not
formally exited. A deterministic G1-humanoid episode can be replayed,
rendered, and scored (D-016, D-018, D-021 — reproducibility now a
regression test) with goal-feasibility/constraint outcomes logged
(`evaluate_goal_graph()`). What's *not* done: a clean separation of
high-level (goal/feasibility) and low-level (skill execution) outcome
logging as originally envisioned — `steps_used`/`wasted_steps` is a proxy,
not that split, since `teleport-on-success` (see
`spikes/task_schema_draft/README.md`) abstracts low-level execution away
almost everywhere. Formal exit also requires the `src/atr/` scaffold and
schema contracts this phase calls for, neither of which exists yet — see
`ai-notes/review-request-task-schema.md` for what's actually gating that.

## Phase 1 — benchmark and oracle

- **Shared:** Implement one task family, controlled instruction grammar,
  interventions, oracle feasibility, constraint checks, and versioned splits.
- **Representation:** Specify observation/trajectory collection requirements and leakage checks.
- **Policy:** Implement simulator hooks, intervention execution, and skill telemetry.

**Exit:** a versioned dataset generator produces leakage-audited splits.

## Phase 2 — static policy baseline

- **Policy:** Train static and oracle-feasibility goal-order-conditioned policies.
- **Representation:** Provide deterministic parsing/encoding and a placeholder belief adapter.
- **Shared:** Quantify nominal performance, adaptation headroom, and post-change failures.

**Exit:** the adaptation gap is large enough to study and not caused by an
unreliable nominal policy.

## Phase 3 — self-supervised representation

- **Shared:** Generate and freeze unlabeled trajectory and evaluation splits.
- **Representation:** Train/compare frozen, fine-tuned, image, temporal, and
  object-centric representations; run diagnostic and feasibility probes.
- **Policy:** Maintain the policy-side adapter and benchmark inference latency.

**Exit:** at least one representation beats declared baselines on held-out data.

## Phase 4 — feasibility inference

- **Representation:** Train per-goal feasibility/change models, calibrate uncertainty,
  implement abstention, and audit shortcuts/counterfactual behavior.
- **Policy:** Evaluate learned beliefs in the oracle-policy scaffold without
  changing policy weights.
- **Shared:** Pass schema, latency, calibration, and end-to-end integration tests.

**Exit:** model beats simple detectors and improves oracle-measured decisions.

## Phase 5 — adaptive policy and intent guard

- **Policy:** Train feasibility-conditioned strategy selection, implement the
  intent guard, and compare modular, monolithic, symbolic, and oracle policies.
- **Representation:** Support representation fine-tuning and run belief-side ablations.
- **Shared:** Diagnose cross-module failures and complete the humanoid integration gate.

**Exit:** improved feasible-goal completion without exceeding the predeclared
intent-violation threshold.

## Phase 6 — generalization and release

- **Representation:** Lead representation, feasibility, calibration, and paraphrase analyses.
- **Policy:** Lead policy, guard, oracle-skill, and humanoid-execution analyses.
- **Shared:** Run held-out tests, complete multi-seed statistics/failure analysis,
  and release the benchmark, configs, permitted checkpoints, tables, and demos.

**Exit:** claims are reproducible from a clean checkout and appropriately scoped.

## Humanoid integration gate

A simpler embodiment may accelerate Phases 1–4, but Phase 5 cannot exit without
running the full visual-language feasibility and intent pipeline on a simulated
humanoid. If humanoid controller instability dominates results, report both
oracle-skill and actual-execution evaluations rather than dropping the humanoid.

## Decision gates

If oracle-feasibility does not improve over the static policy, redesign the task
or adaptation interface before representation work. If visual models succeed
only through leakage, repair the benchmark before scaling. If the guard destroys
goal completion, report the trade-off and revise the constraint representation.
