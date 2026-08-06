"""Tests for atr.policies.domain_randomized (D-065) -- docs/10-evaluation-
and-benchmarks.md's "domain-randomized policy without explicit
feasibility" required baseline, previously unbuilt. Trained against the
canonical env, same as q_learning.py's own first tests
(test_rl_policy.py).

No rendering anywhere here (render_mode=None throughout), same as
test_rl_policy.py -- D-022's rendering bug doesn't apply at this level.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal, feasibility_aware_policy  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.domain_randomized import (  # noqa: E402
    domain_randomized_policy,
    train_domain_randomized_policy,
)
from atr.policies.q_learning import ATTEMPT, SKIP  # noqa: E402

_GRAPH = canonical_example()


def _make_env(intervention_kind: str = "bowl_destroyed", onset_step_range: tuple[int, int] = (5, 15)):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestBlindPolicyLearnsToSkipAnUnreliableGoal:
    """Not assumed -- derived from this project's own reward shape
    (+1.0 achieved, -0.1*steps_used otherwise) and the 50/50
    intervention_kinds default: a goal that's only feasible half the time
    has negative expected value to attempt blindly
    (0.5*1.0 + 0.5*(-0.1*25) = -0.75 versus 0.0 for skipping), so a
    policy with no way to perceive feasibility should learn to skip it
    unconditionally -- confirmed directly on the trained table, not just
    inferred from downstream behavior."""

    def test_always_feasible_goal_learns_attempt(self):
        q = train_domain_randomized_policy(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, n_episodes=120, seed=0,
        )
        assert q["place_mug"][ATTEMPT] > q["place_mug"][SKIP]

    def test_sometimes_infeasible_goal_learns_skip_unconditionally(self):
        q = train_domain_randomized_policy(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, n_episodes=120, seed=0,
        )
        assert q["place_bowl"][SKIP] > q["place_bowl"][ATTEMPT]


class TestBlindPolicyTradesRecallForSafety:
    """The real, measured three-way comparison this baseline exists for:
    unlike feasibility_aware_policy (correctly discriminates), the blind
    policy skips place_bowl unconditionally -- costing it nothing when
    the goal really is infeasible (matches feasibility_aware there), but
    costing it a genuinely achievable goal when it isn't."""

    def test_matches_feasibility_aware_when_goal_is_actually_infeasible(self):
        q = train_domain_randomized_policy(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, n_episodes=120, seed=0,
        )
        results = {}
        for name, run in [
            ("feasibility_aware", feasibility_aware_policy),
            ("domain_randomized", lambda env: domain_randomized_policy(env, q, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()
        assert results["domain_randomized"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["domain_randomized"]["wasted_steps"] == 0

    def test_wrongly_skips_when_goal_was_actually_feasible(self):
        q = train_domain_randomized_policy(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal, n_episodes=120, seed=0,
        )
        results = {}
        for name, run in [
            ("feasibility_aware", feasibility_aware_policy),
            ("domain_randomized", lambda env: domain_randomized_policy(env, q, _GRAPH, attempt_goal, _TRAY_SLOTS)),
        ]:
            env = _make_env(intervention_kind="none")
            try:
                env.reset(seed=0)
                results[name] = run(env)
            finally:
                env.close()
        assert results["feasibility_aware"]["goals_achieved"] == 2
        assert results["domain_randomized"]["goals_achieved"] == 1  # wrongly skipped place_bowl
        assert results["domain_randomized"]["per_goal"]["place_bowl"]["skipped"] is True
