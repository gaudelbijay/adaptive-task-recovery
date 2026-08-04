"""Tests for rl_policy.py -- stage 5 of docs/00-project-overview.md's
build-up order ("replace the scripted/oracle policies with one that's
actually learned"). Same H2 comparison as D-014's original test
(policy_baselines.py), but the feasibility-aware behavior is now discovered
by a tabular Q-learning agent through reward, not hard-coded as a rule.

No rendering anywhere here (render_mode=None throughout) -- D-022's
confirmed upstream rendering bug does not apply to this stage.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import feasibility_aware_policy, static_policy  # noqa: E402
from task_schema_draft.rl_policy import ATTEMPT, SKIP, learned_policy, train_q_table_canonical  # noqa: E402


def _make_env(**kwargs):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos", **kwargs,
    )


class TestQLearningDiscoversTheFeasibilityRule:
    """The rule feasibility_aware_policy hard-codes ("attempt iff feasible")
    should fall out of trial-and-reward training, not be handed to the
    agent -- this checks the actual learned Q-table, not just downstream
    behavior."""

    def test_learns_attempt_when_feasible_skip_when_not(self):
        q = train_q_table_canonical(n_episodes=120, seed=0)
        assert q[("place_mug", True)][ATTEMPT] > q[("place_mug", True)][SKIP]
        assert q[("place_bowl", True)][ATTEMPT] > q[("place_bowl", True)][SKIP]
        assert q[("place_bowl", False)][SKIP] > q[("place_bowl", False)][ATTEMPT]


class TestLearnedPolicyMatchesFeasibilityAwareBehavior:
    def test_same_recall_zero_waste_after_bowl_destroyed(self):
        q = train_q_table_canonical(n_episodes=120, seed=0)
        results = {}
        for name, policy in [
            ("static", static_policy),
            ("feasibility_aware", feasibility_aware_policy),
            ("learned", lambda env: learned_policy(env, q)),
        ]:
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=(2, 3))
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()

        assert results["learned"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["learned"]["wasted_steps"] == 0
        assert results["static"]["wasted_steps"] > 0

    def test_both_goals_achieved_with_no_intervention(self):
        q = train_q_table_canonical(n_episodes=120, seed=0)
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            result = learned_policy(env, q)
        finally:
            env.close()
        assert result["goals_achieved"] == 2
        assert result["wasted_steps"] == 0
