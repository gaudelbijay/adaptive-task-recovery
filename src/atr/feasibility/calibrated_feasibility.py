"""H5 (calibration): D-070 found that a binary feasibility check
(`goal_feasible()`) can't distinguish "safe" from "feasible now but at
risk of becoming infeasible mid-attempt" once intervention timing is wide
enough to span an attempt's own duration.

D-071 corrected D-070's specific causal claim (see `ai-notes/decisions.md`):
this module's first version calibrated a single survival probability per
goal, *pooled* across every `intervention_kind` the env can sample --
exactly the same pooling `train_q_table()`'s `(goal_id, feasible)` state
key does. Measured directly (D-071): that pooled true expected value is
statistically indistinguishable from zero (bootstrap 95% CI straddling
zero, `n=441`), because risk-free `"none"` episodes and genuinely risky
`"bowl_destroyed"` episodes get averaged into one number. What *is*
robustly, confidently negative (CI clearly excludes zero, `n=198`) is the
expected value *conditional on the risky intervention actually being
active*. Calibrating per `(goal_id, intervention_kind)` instead of pooling
recovers that decisive conditional signal directly -- `env.unwrapped.
intervention_kind` is privileged state, at the same privilege level
`goal_feasible()` itself already uses throughout this project (see
`atr.envs.tidy_up_env`), not a new kind of cheating.

`calibrate_survival_probability()` runs real rollouts and empirically
measures, per `(goal_id, intervention_kind)`, P(still achievable through
completion of its own attempt | perceived feasible right now);
`calibrated_feasibility_policy()` reads the episode's actual
`intervention_kind`, looks up the matching calibrated probability, and
attempts a perceived-feasible goal only when doing so has positive
expected value under the same reward shape `train_q_table()`
(`atr.policies.q_learning`) uses (`+1.0` achieved, `-0.1 * steps_used`
otherwise) -- a strict generalization of `feasibility_aware_policy`'s
binary rule (`atr.envs.tidy_up_policies`): it collapses back to "attempt
iff feasible" whenever the calibrated survival probability is high
enough, and grows more conservative as it drops.

Deliberately not reusing Q-learning's own machinery: the point is an
explicit, interpretable probability, calibrated once via direct
Monte-Carlo rollouts (and reported with a bootstrap CI via
`atr.evaluation.harness.bootstrap_ci`, D-042) rather than a small-sample,
recency-biased TD-learned table -- and, unlike a Q-table, it makes
visible whether it was calibrated under a timing distribution that
matches deployment (see the calibration/deployment mismatch test in
`tests/drafts/test_calibrated_feasibility.py`). That is a genuinely
different generalization axis from D-069's held-out-intervention-
*mechanism* result: D-069's generalization was close to free by
construction (the `(goal_id, feasible)` state never encoded mechanism, so
nothing could depend on it); calibration to a timing *distribution* is
not free the same way, and the mismatch test demonstrates that directly
rather than assuming it.
"""

from __future__ import annotations

import random
from typing import Callable

from atr.feasibility.oracle import goal_feasible
from atr.language.goal_graph import GoalGraph
from atr.policies.baselines import _summarize
from atr.policies.q_learning import _REACH_STEPS, _wait

_SUCCESS_REWARD = 1.0
_STEP_COST = 0.1


def calibrate_survival_probability(
    make_env: Callable,
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, str] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 150,
    seed: int = 0,
) -> dict[tuple[str, str], float]:
    """Empirically measures, per `(goal_id, intervention_kind)`, P(actually
    achieved | perceived feasible at this goal's own decision point) via
    real rollouts that always attempt when perceived feasible -- so the
    measurement isn't contaminated by any policy's own skip decisions.
    Keyed by `intervention_kind`, not pooled across it (D-071 -- see
    module docstring for why pooling produces a statistically ambiguous
    number instead of a decisive one). A `(goal_id, intervention_kind)`
    with zero perceived-feasible-then-attempted observations gets 1.0 (no
    evidence of risk -- defaults to the naive "attempt iff feasible"
    assumption rather than an arbitrary pessimistic one)."""
    rng = random.Random(seed)
    feasible_counts: dict[tuple[str, str], int] = {}
    achieved_counts: dict[tuple[str, str], int] = {}

    for _ in range(n_episodes):
        intervention_kind = rng.choice(intervention_kinds)
        onset_step = rng.randint(*onset_step_bounds)
        env = make_env(intervention_kind, (onset_step, onset_step + 1))
        try:
            env.reset(seed=rng.randint(0, 2**31 - 1))
            for i, goal in enumerate(graph.goals):
                feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
                if not feasible:
                    _wait(env, reach_steps)
                    continue
                key = (goal.id, intervention_kind)
                feasible_counts[key] = feasible_counts.get(key, 0) + 1
                result = attempt_goal_fn(env, goal, tray_slots[i], reach_steps)
                if result["achieved"]:
                    achieved_counts[key] = achieved_counts.get(key, 0) + 1
        finally:
            env.close()

    return {key: achieved_counts.get(key, 0) / count for key, count in feasible_counts.items()}


def expected_value_of_attempt(survival_probability: float, reach_steps: int = _REACH_STEPS) -> float:
    """Expected reward of attempting a perceived-feasible goal, matching
    `train_q_table()`'s exact reward shape so a calibrated policy's
    decision threshold (attempt iff this is positive) is directly
    comparable to what a Q-learned policy converges to, not a separately
    invented one."""
    p = survival_probability
    return p * _SUCCESS_REWARD - (1 - p) * _STEP_COST * reach_steps


def calibrated_feasibility_policy(
    env,
    survival_probabilities: dict[tuple[str, str], float],
    graph: GoalGraph,
    attempt_goal_fn: Callable,
    tray_slots: list,
) -> dict:
    """Same result shape as static_policy/feasibility_aware_policy/
    learned_policy for direct comparison. Reads the episode's actual
    `intervention_kind` (privileged state, same level `goal_feasible()`
    already uses) to select the matching calibrated probability, and
    attempts a perceived-feasible goal only when
    `expected_value_of_attempt()` on that probability is positive."""
    intervention_kind = env.unwrapped.intervention_kind
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        p = survival_probabilities.get((goal.id, intervention_kind), 1.0)
        if feasible and expected_value_of_attempt(p) > 0:
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
        else:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
    return _summarize(per_goal)
