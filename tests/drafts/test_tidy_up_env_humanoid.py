"""Tests for the Unitree G1 humanoid version of TidyUp — same goal graph,
oracle feasibility, and intent guard as the panda version
(test_tidy_up_env.py, test_policy_baselines.py, test_intent_guard.py),
different embodiment. See tidy_up_env_humanoid.py's module docstring and
spikes/task_schema_draft/README.md.
"""

import pytest

pytest.importorskip("mani_skill")

import gymnasium as gym  # noqa: E402

import task_schema_draft  # noqa: E402, F401  (registers TidyUpTaskSchemaDraft-Humanoid-v1)
from task_schema_draft.policy_baselines_humanoid import (  # noqa: E402
    feasibility_aware_policy,
    naive_substitution_policy,
    static_policy,
)


def _make_env(**kwargs):
    return gym.make(
        "TidyUpTaskSchemaDraft-Humanoid-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_joint_pos", **kwargs,
    )


class TestTidyUpHumanoidEnv:
    def test_registered(self):
        assert "TidyUpTaskSchemaDraft-Humanoid-v1" in gym.envs.registry

    def test_reset_and_step(self):
        env = _make_env(intervention_kind="none")
        try:
            obs, info = env.reset(seed=0)
            assert "goal_feasibility" in info
            env.step(env.action_space.sample() * 0)
        finally:
            env.close()

    def test_no_false_violations_from_settling(self):
        """Regression test: objects spawn slightly above the kitchen
        counter's real surface and drop a bit while settling. Before the
        settle-window fix, this alone tripped dont_move_glass."""
        env = _make_env(intervention_kind="none")
        try:
            env.reset(seed=0)
            for _ in range(20):
                _, _, _, _, info = env.step(env.action_space.sample() * 0)
            assert not any(info["constraint_violations"].values())
        finally:
            env.close()


class TestHumanoidPolicyComparison:
    def test_static_vs_feasibility_aware_same_recall_less_waste(self):
        results = {}
        for name, policy in [("static", static_policy), ("feasibility_aware", feasibility_aware_policy)]:
            env = _make_env(intervention_kind="bowl_destroyed")
            try:
                env.reset(seed=0)
                results[name] = policy(env)
            finally:
                env.close()
        assert results["static"]["goals_achieved"] == results["feasibility_aware"]["goals_achieved"]
        assert results["feasibility_aware"]["wasted_steps"] == 0
        assert results["static"]["wasted_steps"] > 0

    def test_intent_guard_blocks_substitution_without_recall_cost(self):
        results = {}
        for guarded in (False, True):
            env = _make_env(intervention_kind="bowl_destroyed")
            try:
                env.reset(seed=0)
                results[guarded] = naive_substitution_policy(env, use_intent_guard=guarded)
            finally:
                env.close()
        assert results[False]["dont_move_glass_violated"] is True
        assert results[True]["dont_move_glass_violated"] is False
        assert results[False]["goals_achieved"] == results[True]["goals_achieved"]
