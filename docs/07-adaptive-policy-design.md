---
title: Adaptive Policy and Intent Preservation
status: draft
last_updated: 2026-08-01
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

**A learned variant of #4, closer to what "adaptive" is meant to mean
(2026-08-01):** `spikes/task_schema_draft/rl_policy.py` (D-025) replaces
`feasibility_aware_policy`'s hard-coded "attempt iff feasible" rule with a
tabular Q-learning agent trained on real environment rollouts (~19s, CPU
only) — still conditioned on privileged/oracle feasibility labels, not
learned beliefs, so still #4 in this list rather than #3, but the *policy*
itself is now learned rather than hand-coded. Matches
`feasibility_aware_policy`'s behavior exactly once trained. Still toy scale
(2 goals, 3 meaningful states) — see D-025's own "Consequences" note.

**A second learned variant, for comparison against the first
(2026-08-05, D-060):** `atr.policies.imitation` (promoted from the start —
see D-060 in `ai-notes/decisions.md`) learns the same #4 attempt/skip
decision by behavioral cloning from demonstrations of
`feasibility_aware_policy`'s own rule, instead of Q-learning's
trial-and-reward. Same `(goal_id, feasible) -> {SKIP, ATTEMPT}` state/
action space and `attempt_goal_fn`/`tray_slots` parameterization as
`q_learning.py`, so the two are trained and compared under matched
conditions, not just described side by side.

Where imitation learning is used here, concretely, and where it isn't:
this project's environments already provide a cheap, perfect "expert" —
`goal_feasible()`'s privileged-state rule — so demonstrations cost
nothing to generate (no human teleoperation, no separately-trained
policy to imitate). That's a genuinely different regime from where
imitation learning usually earns its keep (expensive-to-generate reward
signals, or a real robot where trial-and-error exploration is costly or
unsafe) — worth stating plainly rather than overselling the toy result.
The actual finding this comparison produces (D-060, verified, not
assumed):

- Given demonstration coverage comparable to what Q-learning explores
  (both feasible and infeasible states shown), imitation matches
  Q-learning's behavior exactly — the same "recovers the hand-coded
  rule" result D-025 already got for Q-learning, now for a second,
  differently-trained policy.
- Given *narrower* demonstration coverage (e.g. only ever demonstrated
  with the intervention already fired, so the expert is only ever seen
  skipping the bowl goal, never attempting it), imitation inherits that
  narrowness in a way Q-learning's own exploration doesn't: querying a
  state nobody ever demonstrated falls back to a global-majority default
  that, in this exact narrow setup, wrongly abandons a goal that was
  actually achievable. Q-learning, trained with real exploration across
  both feasible and infeasible episodes, doesn't have this gap, since it
  visits the state directly instead of relying on a demonstrator having
  visited it first.

This is the standard, textbook IL-vs-RL coverage trade-off (behavioral
cloning can't correct a demonstration distribution's own gaps; on-policy
exploration can), made concrete and empirically checked in this
project's own toy setting rather than only asserted. A real future
extension, not attempted here: using imitation learning for something
Q-learning's reward signal genuinely *can't* cheaply supervise — e.g.
cloning a scripted or human-teleoperated low-level reach/grasp
trajectory (this project's `attempt_goal_fn` is currently a fixed,
hand-tuned reach, not learned at all), rather than the high-level
attempt/skip decision, where a privileged-state reward is already free.

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

**First toy instance (2026-07-29):** `src/atr/constraints/intent_guard.py`'s
(promoted from `spikes/task_schema_draft/`, D-037)
`validate_action()` is the smallest possible version of this — it rejects a
candidate action targeting a `never_move`-constrained object unless a real
goal in the graph requires touching it, exactly the "unauthorized
replacement" case (substituting the glass for a destroyed bowl). See D-015
in `ai-notes/decisions.md`. Does not yet handle equivalence classes,
ordering/dependency edges, or the harder recall/safety trade-off this
section's design implies.

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
