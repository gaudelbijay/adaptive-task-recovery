"""Stage 5 of docs/00-project-overview.md's build-up order, thin env-
specific wrapper. The actual Q-learning algorithm (`train_q_table()`,
`learned_policy()`) was promoted to `atr.policies.q_learning` (D-041) --
this file just supplies the canonical tabletop env's own
`attempt_goal`/`_TRAY_SLOTS`/`canonical_example()`, the same relationship
`atr.envs.tidy_up_policies`'s wrappers have to `atr.policies.baselines`
(D-040/D-046).
"""

from __future__ import annotations

import gymnasium as gym

from atr.language.goal_graph import GoalGraph, canonical_example
from atr.policies.q_learning import (
    ATTEMPT,
    SKIP,
    learned_policy as _learned_policy,
    train_q_table,
)
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal

__all__ = ["ATTEMPT", "SKIP", "train_q_table", "train_q_table_canonical", "learned_policy"]


def _make_canonical_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def train_q_table_canonical(n_episodes: int = 120, seed: int = 0) -> dict:
    """train_q_table() against the canonical tabletop env (tidy_up_env.py) --
    the specific instance D-025 originally built and tested."""
    return train_q_table(
        make_env=_make_canonical_env, graph=canonical_example(), tray_slots=_TRAY_SLOTS,
        attempt_goal_fn=attempt_goal, n_episodes=n_episodes, seed=seed,
    )


def learned_policy(env, q_table: dict, graph: GoalGraph = None) -> dict:
    """learned_policy() (atr.policies.q_learning) against the canonical
    tabletop env."""
    return _learned_policy(env, q_table, graph or canonical_example(), attempt_goal, _TRAY_SLOTS)
