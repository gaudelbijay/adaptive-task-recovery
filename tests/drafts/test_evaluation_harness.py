"""Tests for atr.evaluation.harness (D-042) -- the first real
implementation of docs/10-evaluation-and-benchmarks.md's statistical
protocol (paired seeds, bootstrap confidence intervals), not just a
single seed=0 comparison the way every prior policy comparison in this
project was run.
"""

import gymnasium as gym
import pytest

from atr.evaluation.harness import bootstrap_ci, compare_policies, run_episode


class TestBootstrapCi:
    def test_constant_values_have_a_zero_width_interval(self):
        mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0, 1.0])
        assert mean == lo == hi == 1.0

    def test_interval_contains_the_mean(self):
        mean, lo, hi = bootstrap_ci([0.0, 0.0, 1.0, 1.0, 1.0], seed=1)
        assert lo <= mean <= hi

    def test_wider_ci_is_a_superset_of_narrower_ci(self):
        values = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        _, lo90, hi90 = bootstrap_ci(values, ci=0.90, seed=0)
        _, lo99, hi99 = bootstrap_ci(values, ci=0.99, seed=0)
        assert lo99 <= lo90
        assert hi99 >= hi90

    def test_deterministic_given_seed(self):
        values = [0.0, 1.0, 2.0, 1.0, 0.0, 2.0]
        result_a = bootstrap_ci(values, seed=7)
        result_b = bootstrap_ci(values, seed=7)
        assert result_a == result_b


pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from task_schema_draft.policy_baselines import feasibility_aware_policy, static_policy  # noqa: E402
from task_schema_draft.rl_policy import learned_policy, train_q_table_canonical  # noqa: E402


def _make_env():
    return gym.make(
        "TidyUpTaskSchemaDraft-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind="bowl_destroyed", onset_step_range=(2, 3),
    )


class TestRunEpisode:
    def test_returns_the_policy_result_shape(self):
        result = run_episode(_make_env, static_policy, seed=0)
        assert "goals_achieved" in result
        assert "wasted_steps" in result


class TestComparePoliciesOnCanonicalEnv:
    """The real deliverable: H2's static-vs-feasibility-aware claim
    (D-014), run with docs/10's actual statistical protocol -- paired
    seeds, bootstrap CIs -- for the first time, instead of a single
    seed=0 point comparison."""

    def test_feasibility_aware_and_learned_have_disjoint_wasted_steps_ci_from_static(self):
        q_table = train_q_table_canonical(n_episodes=120, seed=0)
        policies = {
            "static": static_policy,
            "feasibility_aware": feasibility_aware_policy,
            "learned": lambda env: learned_policy(env, q_table),
        }
        report = compare_policies(_make_env, policies, seeds=list(range(30)), n_resamples=1000)

        static_wasted_lo = report["static"]["wasted_steps"][1]
        for name in ("feasibility_aware", "learned"):
            _, _, hi = report[name]["wasted_steps"]
            assert hi < static_wasted_lo, (
                f"{name}'s wasted_steps CI should not overlap static's: "
                f"{name}={report[name]['wasted_steps']}, static={report['static']['wasted_steps']}"
            )

        # H2's other half: no completion cost for that efficiency gain.
        for name in ("feasibility_aware", "learned"):
            static_mean = report["static"]["goals_achieved"][0]
            name_mean = report[name]["goals_achieved"][0]
            assert name_mean == pytest.approx(static_mean, abs=0.15)
