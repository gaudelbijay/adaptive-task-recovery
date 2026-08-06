"""Domain-randomized policy without explicit feasibility (D-065) --
docs/10-evaluation-and-benchmarks.md's required-baselines list names this
explicitly, distinct from every other policy in this project: every one
of them (`baselines.py`'s static/oracle-feasibility/naive-substitution,
`q_learning.py`'s learned policy, `imitation.py`'s behavioral cloning)
either hard-codes a feasibility rule or is trained/demonstrated *with* a
feasibility signal as part of its state. This is the ablation of that --
same domain randomization `q_learning.train_q_table()` already trains
under (intervention kind and onset timing varied every episode), but the
state key drops the feasibility bit entirely: `goal_id -> {SKIP, ATTEMPT}`,
not `(goal_id, feasible) -> {SKIP, ATTEMPT}`. The policy has no way to
perceive whether the current episode's goal is actually feasible, only
which goal it's looking at.

What this is expected to (and does) discover, not assumed: with
`intervention_kinds=("none", "bowl_destroyed")` at 50/50 and this
project's existing reward shape (`+1.0` achieved, `-0.1 * steps_used`
otherwise), the expected value of attempting averages over both outcomes
-- roughly `0.5*(+1.0) + 0.5*(-0.1*25) = -0.75` for a goal that's
sometimes infeasible, versus `0.0` for skipping. A blind policy should
therefore learn to skip goals it can't reliably complete, converging to
something more conservative than `static_policy` (always attempts) but
less accurate than `feasibility_aware_policy`/`learned_policy` (skips
indiscriminately, including the episodes where the goal actually was
fine) -- a real three-way tradeoff, not assumed here, verified in
`tests/drafts/test_domain_randomized.py`.
"""

from __future__ import annotations

import random
from typing import Callable

import gymnasium as gym

from atr.language.goal_graph import GoalGraph
from atr.policies.baselines import _summarize
from atr.policies.q_learning import ATTEMPT, SKIP, _wait, _REACH_STEPS


def train_domain_randomized_policy(
    make_env: Callable[[str, tuple[int, int]], "gym.Env"],
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, ...] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 120,
    seed: int = 0,
) -> dict:
    """Tabular Q-learning over `goal_id -> {SKIP: q, ATTEMPT: q}` only --
    same domain-randomized training loop as `q_learning.train_q_table()`
    (same env/randomization plumbing, same reward shape), minus the
    feasibility bit in the state key. Deliberately duplicates
    `train_q_table()`'s loop rather than adding a `blind: bool` flag to
    it -- the two functions have different state spaces entirely (a
    `dict[str, ...]` here vs. `dict[tuple[str, bool], ...]` there), and
    conditionally branching one function's key type on a flag would be
    more confusing than two small, clearly-named functions sharing the
    same env-interaction helpers (`_wait`, `SKIP`/`ATTEMPT`)."""
    rng = random.Random(seed)
    q: dict[str, dict[int, float]] = {}

    for ep in range(n_episodes):
        epsilon = max(0.05, 1.0 - ep / (n_episodes * 0.6))
        intervention_kind = rng.choice(intervention_kinds)
        onset_step = rng.randint(*onset_step_bounds)
        env = make_env(intervention_kind, (onset_step, onset_step + 1))
        try:
            env.reset(seed=rng.randint(0, 2**31 - 1))
            for i, goal in enumerate(graph.goals):
                key = goal.id
                q.setdefault(key, {SKIP: 0.0, ATTEMPT: 0.0})

                if rng.random() < epsilon:
                    action = rng.choice([SKIP, ATTEMPT])
                else:
                    action = max(q[key], key=q[key].get)

                if action == SKIP:
                    reward = 0.0
                    _wait(env, reach_steps)
                else:
                    result = attempt_goal_fn(env, goal, tray_slots[i], reach_steps)
                    reward = 1.0 if result["achieved"] else -0.1 * result["steps_used"]

                q[key][action] += 0.3 * (reward - q[key][action])
        finally:
            env.close()
    return q


def domain_randomized_policy(
    env, q_table: dict, graph: GoalGraph, attempt_goal_fn: Callable, tray_slots: list,
) -> dict:
    """Runs the greedy policy from a table trained by
    `train_domain_randomized_policy()`. Same result shape as
    `static_policy`/`feasibility_aware_policy`/`learned_policy`, for
    direct comparison -- deliberately does NOT read
    `env.unwrapped._world_state()`/`goal_feasible()` anywhere in this
    function, since that's exactly the signal this baseline is testing
    the absence of."""
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        actions = q_table.get(goal.id, {SKIP: 0.0, ATTEMPT: 0.0})
        action = max(actions, key=actions.get)
        if action == SKIP:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
        else:
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
    return _summarize(per_goal)
