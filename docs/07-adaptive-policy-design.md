---
title: Adaptive Policy and Intent Preservation
status: draft
last_updated: 2026-07-26
---

# Adaptive Policy and Intent Preservation

## Decision problem

The agent receives visual history and a language-derived goal graph. At each
step it updates feasibility beliefs and chooses an action or subgoal. The policy
should maximize achieved goal value while penalizing wasted effort and obeying
hard constraints.

One operational objective is:

```text
maximize  E[sum_i priority_i * achieved_i - step_cost - invalid_substitution]
subject to P(any hard-constraint violation) <= epsilon
```

This objective must be evaluated separately from the reward used in training to
avoid defining success circularly.

## Policy variants

1. **Static language-conditioned policy:** never receives explicit change or
   feasibility estimates.
2. **Change-aware policy:** receives a change embedding but no per-goal labels.
3. **Feasibility-conditioned policy:** receives learned per-goal beliefs.
4. **Oracle-feasibility policy:** receives privileged labels and establishes
   adaptation headroom.
5. **Symbolic replanner:** uses privileged or learned state with explicit
   preconditions and effects.
6. **Monolithic adaptive policy:** matched capacity without modular heads.

**First toy instances of #1 and #4 (2026-07-29):** `spikes/task_schema_draft/policy_baselines.py`
implements `static_policy` (#1) and `feasibility_aware_policy` (#4 —
privileged/oracle labels, not learned beliefs) and runs them against the
docs/01 worked example. Result: after an irreversible change, both achieve
the same goals, but the oracle-feasibility policy wastes zero steps versus
the static policy's wasted attempt on the now-infeasible goal (D-014 in
`ai-notes/decisions.md`). Toy scale only — see that file's scope notes. This
does not yet establish the "adaptation headroom" #4 is meant to define; it's
a single hand-authored scenario, not a swept evaluation.

## Strategy adaptation

The initial implementation should use a high-level policy that selects the next
goal or semantic skill, with a humanoid low-level controller executing it. This
makes abandonment, substitution, ordering, and constraint violations observable
without requiring the research policy to relearn balance and grasp control.
Skill results—including retryable failure, safety stop, and capability limits—
return to the high-level belief state. An end-to-end action policy remains a
possible later baseline, not a prerequisite.

## Intent model

Compile controlled natural language into:

- positive goal predicates;
- hard negative constraints;
- ordering and dependency edges;
- object identity and allowed equivalence classes;
- priorities/preferences when stated.

The intent guard validates candidate high-level choices and, where possible,
primitive actions. It must distinguish an explicitly allowed substitute from a
semantically convenient but unauthorized replacement.

## Uncertainty and abstention

When evidence is ambiguous, the agent may gather information, delay commitment,
or stop safely. Information-seeking actions must incur cost so abstention does
not become a loophole. Compare forced decisions, calibrated thresholds, and a
learned information-gathering strategy.

## Reward safeguards

- Never reward the agent for merely predicting a goal is infeasible.
- Score goal completion against the original instruction and oracle state.
- Apply hard violation costs independently of the learned language model.
- Track abandonment of feasible goals as a distinct error.
- Test adversarial cases where violating intent would yield higher task reward.
