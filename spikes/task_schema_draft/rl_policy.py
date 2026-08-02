"""Stage 5 of docs/00-project-overview.md's build-up order: replace the
scripted/oracle policies with one that's actually learned via reward,
instead of hand-coded. `feasibility_aware_policy` in policy_baselines.py
always implemented "attempt iff feasible" as a hard-coded rule; this stage
asks whether an agent can discover that same rule from trial and reward,
without ever being told it.

Deliberately narrow scope: this is tabular Q-learning over the *goal-attempt
decision* (attempt vs. skip, given a feasibility observation) -- not a
learned low-level motor policy. Low-level control (the reach phase) is
exactly `attempt_goal()` from policy_baselines.py, unchanged; only the
decision of *whether* to attempt is learned. This matches the project's own
scope throughout -- the research question is strategy/feasibility
reasoning, not motor control (see D-024's teleport-on-success note, and
docs/00's "Out of scope for v1": training low-level control from scratch).

No rendering anywhere in this file (render_mode=None throughout) -- this
stays at the privileged-state level, same as D-014's original H2 test, so
D-022's confirmed upstream rendering bug never comes into play here.

The state a decision is keyed on is (goal_id, feasible) -- `feasible` comes
from the same privileged-state `goal_feasible()` query
`feasibility_aware_policy` already uses directly as a rule. The point isn't
that the agent discovers feasibility from scratch (that's stage 3/4's job,
clip_feasibility.py / dinov2_probe.py) -- it's that, given the same input a human
already hand-coded a rule for, a Q-learning agent trained purely on reward
recovers the same rule on its own, across randomized episodes where
sometimes attempting is right and sometimes skipping is.

`train_q_table()` is env-agnostic on purpose (D-030): it originally existed
twice, once here specific to the canonical tabletop env and once again in
end_to_end.py specific to the ReplicaCAD-humanoid env, differing only in
which env/goals/attempt function got passed in. Same algorithm, same
`_wait()` timing-consistency fix needed in both places -- worth one
parameterized function, not two near-duplicates that could silently drift
apart.
"""

from __future__ import annotations

import random
from typing import Callable

import gymnasium as gym
import numpy as np

from task_schema_draft.goal_graph import GoalGraph, canonical_example
from task_schema_draft.oracle_feasibility import goal_feasible
from task_schema_draft.policy_baselines import _TRAY_SLOTS, _summarize, attempt_goal

SKIP, ATTEMPT = 0, 1
_ALPHA = 0.3  # single-step decisions per goal -- no bootstrapping needed, so no gamma/discount
_REACH_STEPS = 25  # matches attempt_goal()'s default -- see _wait() below for why this matters


def _make_canonical_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    return gym.make(
        "TidyUpTaskSchemaDraft-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def _wait(env, steps: int = _REACH_STEPS):
    """Holds position for `steps` control steps -- same elapsed-time cost as
    an attempt, without actually reaching. Real bug found building this: the
    existing baselines in policy_baselines.py always attempt the first goal
    unconditionally (it's always feasible), so the second goal's feasibility
    check always happens after the same fixed elapsed time, and the
    intervention (fixed onset_step) has always already fired by then. A
    Q-learning agent can explore SKIP on the first goal too, which shortens
    elapsed time before the second goal's feasibility check -- occasionally
    reading "feasible" correctly at check-time, then having the intervention
    fire mid-attempt, achieving False despite a True feasibility label. That
    was a real, systematic source of negative reward bias during training
    (confirmed: caused the ("place_bowl", True) Q-value to converge negative
    at n_episodes=120), not just noise -- fixed by keeping elapsed time
    consistent regardless of which action is taken. Hit again, independently,
    building end_to_end.py's training loop before this function existed
    (D-029) -- one more reason to have exactly one implementation."""
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        env.step(zero_action)


def train_q_table(
    make_env: Callable[[str, tuple[int, int]], "gym.Env"],
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable = attempt_goal,
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

    `make_env(intervention_kind, onset_step_range) -> env` and `graph` let
    this run against any TidyUp env/goal-graph combination -- see module
    docstring for why this is one parameterized function, not one per env.
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


def train_q_table_canonical(n_episodes: int = 120, seed: int = 0) -> dict:
    """train_q_table() against the canonical tabletop env (tidy_up_env.py) --
    the specific instance D-025 originally built and tested."""
    return train_q_table(
        make_env=_make_canonical_env, graph=canonical_example(), tray_slots=_TRAY_SLOTS,
        n_episodes=n_episodes, seed=seed,
    )


def learned_policy(env, q_table: dict, graph: GoalGraph = None) -> dict:
    """Runs the greedy (argmax) policy from a trained Q-table. Same result
    shape as static_policy/feasibility_aware_policy for direct comparison."""
    graph = graph or canonical_example()
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        key = (goal.id, feasible)
        actions = q_table.get(key, {SKIP: 0.0, ATTEMPT: 0.0})
        action = max(actions, key=actions.get)
        if action == SKIP:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
        else:
            per_goal[goal.id] = attempt_goal(env, goal, _TRAY_SLOTS[i])
    return _summarize(per_goal)
