"""Env-agnostic tabular Q-learning over the goal-attempt decision (D-025,
promoted D-041), parameterized the same way `baselines.py` is (D-040):
`attempt_goal_fn`/`tray_slots` supplied by whichever spike env is
training, not hardcoded to one.

Deliberately narrow scope: this learns the *goal-attempt decision*
(attempt vs. skip, given a feasibility observation) -- not a learned
low-level motor policy. Low-level control (the reach phase) is exactly
whatever `attempt_goal_fn` the caller supplies, unchanged; only the
decision of *whether* to attempt is learned. Matches this project's scope
throughout -- the research question is strategy/feasibility reasoning,
not motor control.

No rendering anywhere in this file (render_mode=None throughout in every
caller) -- this stays at the privileged-state level, so D-022's confirmed
upstream rendering bug never comes into play here.

The state a decision is keyed on is (goal_id, feasible) -- `feasible`
comes from the same privileged-state `goal_feasible()` query
`feasibility_aware_policy` (baselines.py) already uses directly as a
rule. The point isn't that the agent discovers feasibility from scratch
(that's a perceptual-feasibility model's job, e.g. clip_feasibility.py)
-- it's that, given the same input a human already hand-coded a rule
for, a Q-learning agent trained purely on reward recovers the same rule
on its own.

Promoted here already env-agnostic on purpose (D-030, before this
promotion existed): this function originally existed twice, once
specific to the canonical tabletop env and once again specific to the
ReplicaCAD-humanoid env, differing only in which env/goals/attempt
function got passed in. Same algorithm, same `_wait()` timing-consistency
fix needed in both places -- worth one parameterized function, not two
near-duplicates that could silently drift apart (exactly the failure
mode D-040 later confirmed for a different pair of duplicates).
`attempt_goal_fn` has no default here (unlike its pre-promotion form,
which defaulted to the canonical env's `attempt_goal` -- a coupling that
would have pointed promoted code back at spike code); every caller
supplies its own.
"""

from __future__ import annotations

import random
from typing import Callable

import gymnasium as gym
import numpy as np

from atr.feasibility.oracle import goal_feasible
from atr.language.goal_graph import GoalGraph
from atr.policies.baselines import _summarize

SKIP, ATTEMPT = 0, 1
_ALPHA = 0.3  # single-step decisions per goal -- no bootstrapping needed, so no gamma/discount
_REACH_STEPS = 25  # matches every current attempt_goal_fn's own default -- see _wait() below for why this matters


def greedy_action(q_table: dict, key: tuple) -> int:
    """argmax over a trained Q-table's two actions for `key`, defaulting to
    indifferent (0.0, 0.0) for an unseen key. Pulled out (D-050) because
    learned_policy() below and pipeline.py's run_end_to_end_episode() had
    this exact lookup duplicated -- same decision rule, applied to two
    different feasibility signals (privileged state vs. a rendered
    frame), which is the actual point of run_end_to_end_episode() and
    worth keeping distinct; the *lookup* itself has no reason to be
    written twice."""
    actions = q_table.get(key, {SKIP: 0.0, ATTEMPT: 0.0})
    return max(actions, key=actions.get)


def _wait(env, steps: int = _REACH_STEPS):
    """Holds position for `steps` control steps -- same elapsed-time cost as
    an attempt, without actually reaching. Real bug found building this: a
    deterministic baseline that always attempts the first goal
    unconditionally (it's always feasible) makes the second goal's
    feasibility check always happen after the same fixed elapsed time, and
    the intervention (fixed onset_step) has always already fired by then. A
    Q-learning agent can explore SKIP on the first goal too, which shortens
    elapsed time before the second goal's feasibility check -- occasionally
    reading "feasible" correctly at check-time, then having the intervention
    fire mid-attempt, achieving False despite a True feasibility label. That
    was a real, systematic source of negative reward bias during training
    (confirmed: caused a Q-value for a feasible goal to converge negative),
    not just noise -- fixed by keeping elapsed time consistent regardless of
    which action is taken. Hit twice, independently, before being unified
    into this one function (D-030) -- one more reason to have exactly one
    implementation."""
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        env.step(zero_action)


def train_q_table(
    make_env: Callable[[str, tuple[int, int]], "gym.Env"],
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, str] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 120,
    seed: int = 0,
) -> dict:
    """Tabular Q-learning over (goal_id, feasible) -> {SKIP: q, ATTEMPT: q}.
    Trained across randomized episodes (intervention present or not, timing
    varied) so "attempt iff feasible" has to be discovered from reward, not
    handed to the agent.

    `make_env(intervention_kind, onset_step_range) -> env`, `graph`,
    `attempt_goal_fn`, and `tray_slots` let this run against any TidyUp
    env/goal-graph combination -- see module docstring for why this is one
    parameterized function, not one per env.
    """
    rng = random.Random(seed)
    q: dict[tuple[str, bool], dict[int, float]] = {}

    for ep in range(n_episodes):
        epsilon = max(0.05, 1.0 - ep / (n_episodes * 0.6))
        intervention_kind = rng.choice(intervention_kinds)
        onset_step = rng.randint(*onset_step_bounds)
        env = make_env(intervention_kind, (onset_step, onset_step + 1))
        try:
            env.reset(seed=rng.randint(0, 2**31 - 1))
            for i, goal in enumerate(graph.goals):
                feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
                key = (goal.id, feasible)
                q.setdefault(key, {SKIP: 0.0, ATTEMPT: 0.0})

                if rng.random() < epsilon:
                    action = rng.choice([SKIP, ATTEMPT])
                else:
                    action = max(q[key], key=q[key].get)

                if action == SKIP:
                    reward = 0.0
                    _wait(env, reach_steps)  # keeps elapsed time consistent -- see _wait()'s docstring
                else:
                    result = attempt_goal_fn(env, goal, tray_slots[i], reach_steps)
                    reward = 1.0 if result["achieved"] else -0.1 * result["steps_used"]

                q[key][action] += _ALPHA * (reward - q[key][action])
        finally:
            env.close()
    return q


def learned_policy(
    env, q_table: dict, graph: GoalGraph, attempt_goal_fn: Callable, tray_slots: list,
) -> dict:
    """Runs the greedy (argmax) policy from a trained Q-table. Same result
    shape as static_policy/feasibility_aware_policy (baselines.py) for
    direct comparison. `attempt_goal_fn`/`tray_slots`: same parameterization
    as train_q_table(), no longer hardcoded to one env (a pre-promotion
    inconsistency within this same module -- train_q_table() was already
    parameterized, this wasn't)."""
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        key = (goal.id, feasible)
        action = greedy_action(q_table, key)
        if action == SKIP:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
        else:
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
    return _summarize(per_goal)
