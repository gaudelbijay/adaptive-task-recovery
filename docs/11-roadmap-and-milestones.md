---
title: Roadmap and Milestones
status: draft
last_updated: 2026-07-26
---

# Roadmap and Milestones

## Phase 0 — foundations

- Scaffold the repository and pin dependencies.
- Compare humanoid-capable environments with a minimal visual-language task.
- Import a humanoid asset and validate reusable low-level skill interfaces.
- Define goal, constraint, intervention, and evaluation schemas.

**Exit:** one deterministic humanoid episode can be replayed, rendered, and
scored, with high- and low-level outcomes logged separately.

## Phase 1 — benchmark and oracle

- Implement one task family and controlled instruction grammar.
- Add irreversible, reversible, and neutral interventions.
- Implement and test oracle goal feasibility and constraint checking.

**Exit:** a versioned dataset generator produces leakage-audited splits.

## Phase 2 — static policy baseline

- Train a language-conditioned policy without adaptation machinery.
- Quantify nominal performance and post-change failure modes.

**Exit:** the adaptation gap is large enough to study and not caused by an
unreliable nominal policy.

## Phase 3 — self-supervised representation

- Collect unlabeled visual trajectories.
- Train/compare image, temporal, and object-centric objectives.
- Run diagnostic probes and downstream feasibility tests.

**Exit:** at least one representation beats declared baselines on held-out data.

## Phase 4 — feasibility inference

- Train per-goal feasibility and change models.
- Calibrate uncertainty and implement abstention.
- Audit shortcuts and counterfactual behavior.

**Exit:** model beats simple detectors and improves oracle-measured decisions.

## Phase 5 — adaptive policy and intent guard

- Train feasibility-conditioned strategy selection.
- Implement constraint shielding and invalid-substitution checks.
- Compare modular, monolithic, symbolic, and oracle variants.

**Exit:** improved feasible-goal completion without exceeding the predeclared
intent-violation threshold.

## Phase 6 — generalization and release

- Run held-out intervention, composition, layout, object, and paraphrase tests.
- Complete multi-seed statistics and failure analysis.
- Release benchmark generator, configs, checkpoints where licensing permits,
  result tables, and demos.

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
