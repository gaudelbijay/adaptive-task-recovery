"""Env-agnostic imitation learning (behavioral cloning) over the same
goal-attempt decision `atr.policies.q_learning` learns via reward (D-060) --
same `(goal_id, feasible) -> {SKIP, ATTEMPT}` state/action space, same
`attempt_goal_fn`/`tray_slots` parameterization pattern as
`baselines.py`/`q_learning.py`, so the two can be trained and compared
under genuinely comparable conditions, not just described side by side.

Where this is meant to be used: docs/07-adaptive-policy-design.md frames
the adaptive policy as a research axis, not a single fixed algorithm --
this is the second concrete instance alongside Q-learning (D-025/D-041),
giving H2's "does an adaptive strategy improve on static" question a
second, differently-trained way to reach the same decision, and a real
place to study *how* two learning paradigms differ, not just whether
either works. `docs/00-project-overview.md`'s stage 5 ("replace the
scripted/oracle policies with one that's actually learned") already
covers Q-learning; this is a second real instance of the same stage, not
a new one.

The actual comparison this module exists to support (see
tests/drafts/test_imitation_policy.py): given demonstrations that cover
the same states Q-learning explores (both `feasible=True` and
`feasible=False` shown), imitation matches Q-learning's behavior exactly
-- same toy-scale result D-025 already got for Q-learning vs. the
hand-coded rule. Given *narrow* demonstrations (e.g. only ever
demonstrated with the intervention firing, so the expert is only ever
seen skipping, never attempting), imitation inherits that narrowness --
it has no analogue of Q-learning's own exploration, which is exactly
what let Q-learning encounter states nobody ever demonstrated. This is
not a novel research claim; it is the textbook IL-vs-RL coverage
trade-off (behavioral cloning has no way to correct a demonstration
distribution's own gaps; on-policy reward-driven exploration does),
made concrete and testable in this project's own toy setting rather than
just asserted.

The "expert" demonstrated here is `feasibility_aware_policy`'s own rule
(attempt iff feasible, from `atr.feasibility.oracle.goal_feasible()`) --
the same rule D-014 hard-coded and D-025 later showed Q-learning
recovers independently. Using it as the imitation-learning teacher too
keeps the three-way comparison (hard-coded / Q-learned / imitation-
learned) apples-to-apples: same ground-truth rule, three different ways
of arriving at it (or failing to, depending on demonstration coverage).
"""

from __future__ import annotations

import random
from typing import Callable

import gymnasium as gym

from atr.feasibility.oracle import goal_feasible
from atr.language.goal_graph import GoalGraph
from atr.policies.baselines import _summarize
from atr.policies.q_learning import ATTEMPT, SKIP, _wait, _REACH_STEPS

Demonstration = tuple[tuple[str, bool], int]


def collect_demonstrations(
    make_env: Callable[[str, tuple[int, int]], "gym.Env"],
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, ...] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 40,
    seed: int = 0,
) -> list[Demonstration]:
    """Rolls out `n_episodes` with the expert rule (attempt iff feasible,
    via `goal_feasible()` -- the same privileged-state check
    `feasibility_aware_policy` uses directly) deciding every action,
    recording each `((goal_id, feasible), action)` pair actually taken.

    Same env/randomization plumbing as `q_learning.train_q_table()` (make_env,
    onset_step_bounds, reach_steps) so a demonstration set and a Q-table
    can be trained under comparable conditions. `intervention_kinds`
    controls demonstration *coverage* on purpose -- pass a narrower tuple
    (e.g. just `("bowl_destroyed",)`) to build a demonstration set that
    never shows the expert attempting a feasible goal, the deliberately
    limited-coverage case this module's docstring describes."""
    rng = random.Random(seed)
    demonstrations: list[Demonstration] = []

    for _ in range(n_episodes):
        intervention_kind = rng.choice(intervention_kinds)
        onset_step = rng.randint(*onset_step_bounds)
        env = make_env(intervention_kind, (onset_step, onset_step + 1))
        try:
            env.reset(seed=rng.randint(0, 2**31 - 1))
            for i, goal in enumerate(graph.goals):
                feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
                key = (goal.id, feasible)
                action = ATTEMPT if feasible else SKIP
                demonstrations.append((key, action))
                if action == SKIP:
                    _wait(env, reach_steps)
                else:
                    attempt_goal_fn(env, goal, tray_slots[i], reach_steps)
        finally:
            env.close()
    return demonstrations


def train_bc_table(demonstrations: list[Demonstration]) -> dict:
    """Behavioral cloning: for each state key seen in `demonstrations`,
    predicts the majority demonstrated action there -- standard
    frequency-based imitation learning, the supervised-learning analogue
    of `train_q_table()`'s reward-driven table.

    A key never demonstrated at all falls back to the *global* majority
    action across every demonstration, any key -- the standard default
    for an unseen class in a frequency classifier, not an arbitrary
    choice made to force a particular result. If every demonstration
    happens to show ATTEMPT (e.g. a demonstration set with no
    infeasible-state coverage at all), that bias is the demonstration
    set's own, surfaced here rather than hidden by a hand-picked default
    in the other direction."""
    per_key_counts: dict[tuple[str, bool], dict[int, int]] = {}
    global_counts = {SKIP: 0, ATTEMPT: 0}
    for key, action in demonstrations:
        per_key_counts.setdefault(key, {SKIP: 0, ATTEMPT: 0})
        per_key_counts[key][action] += 1
        global_counts[action] += 1

    bc_table = {key: max(counts, key=counts.get) for key, counts in per_key_counts.items()}
    bc_table["__default__"] = max(global_counts, key=global_counts.get)
    return bc_table


def bc_action(bc_table: dict, key: tuple[str, bool]) -> int:
    """Looks up `key` in a trained BC table, falling back to
    `train_bc_table()`'s recorded global-majority default for a key that
    was never demonstrated."""
    return bc_table.get(key, bc_table["__default__"])


def imitation_policy(
    env, bc_table: dict, graph: GoalGraph, attempt_goal_fn: Callable, tray_slots: list,
) -> dict:
    """Runs the greedy behavioral-cloning policy from a trained BC table.
    Same result shape as `static_policy`/`feasibility_aware_policy`
    (baselines.py) and `learned_policy()` (q_learning.py), for direct
    comparison."""
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        key = (goal.id, feasible)
        action = bc_action(bc_table, key)
        if action == SKIP:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
        else:
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
    return _summarize(per_goal)
