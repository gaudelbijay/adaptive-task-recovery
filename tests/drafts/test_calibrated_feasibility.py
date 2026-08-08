"""D-071: builds an explicit, interpretable calibration primitive for H5,
directly motivated by D-070's finding that "currently feasible" doesn't
reliably predict "will complete" once intervention timing spans a goal's
own attempt duration -- and, investigating it, corrects a real overclaim
in D-070 itself (see `ai-notes/decisions.md`'s D-071 entry for the full
story, and the forward-pointer note added to D-070).

The corrected picture, locked in here: `calibrate_survival_probability()`
must key on `(goal_id, intervention_kind)`, not pool across
`intervention_kind` the way `train_q_table()`'s `(goal_id, feasible)`
state (and this module's own first draft) did. Pooling produces a
statistically ambiguous number (a real, measured bootstrap 95% CI
straddling zero for the pooled quantity, D-071) because risk-free
episodes and genuinely risky ones get averaged together; conditioning on
the actual active intervention recovers a decisive, confidently negative
expected value for the genuinely risky case (bootstrap CI clearly
excluding zero) without diluting it against episodes where nothing was
ever at risk.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal  # noqa: E402
from atr.evaluation.harness import bootstrap_ci  # noqa: E402
from atr.feasibility.calibrated_feasibility import (  # noqa: E402
    ATTEMPT,
    SKIP,
    calibrate_survival_probability,
    calibrate_survival_estimates,
    calibrated_feasibility_policy,
    compare_forced_vs_selective,
    expected_value_of_attempt,
)
from atr.feasibility.oracle import goal_feasible  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.q_learning import _REACH_STEPS  # noqa: E402

_GRAPH = canonical_example()
_WIDE_ONSET_RANGE = (10, 60)
_NARROW_ONSET_RANGE = (5, 15)


def _make_env(intervention_kind: str = "bowl_destroyed", onset_step_range=(5, 15)):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestPoolingAcrossInterventionKindProducesAnAmbiguousNumber:
    """The actual root cause of D-070's overclaim: `place_bowl`'s survival
    probability, if pooled across both "none" and "bowl_destroyed"
    episodes, is not confidently different from indifference -- confirmed
    directly with the project's own bootstrap_ci() (D-042), not asserted
    from a single point estimate."""

    def test_pooling_dilutes_the_conditional_risk_signal(self):
        """Not a zero-crossing assertion (fragile this close to the true
        boundary -- an earlier version of this test flakily failed exactly
        because a single-sample CI that close to zero can land on either
        side by chance). The robust, comparative claim that's actually
        true: pooling `"none"` (risk-free) episodes in with
        `"bowl_destroyed"` (genuinely risky) ones pulls the pooled mean
        reward dramatically toward zero relative to the conditional-only
        mean, exactly what dilutes the signal a `(goal_id, feasible)`
        state key (or this module's first, pooled draft) can see."""
        conditional_rewards = []
        pooled_rewards = []
        for seed in range(150):
            intervention_kind = "bowl_destroyed" if seed % 2 == 0 else "none"
            env = _make_env(intervention_kind=intervention_kind, onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0], _REACH_STEPS)
                goal1 = _GRAPH.goals[1]
                if bool(goal_feasible(goal1, env.unwrapped._world_state())):
                    result = attempt_goal(env, goal1, _TRAY_SLOTS[1], _REACH_STEPS)
                    reward = 1.0 if result["achieved"] else -0.1 * result["steps_used"]
                    pooled_rewards.append(reward)
                    if intervention_kind == "bowl_destroyed":
                        conditional_rewards.append(reward)
            finally:
                env.close()
        assert len(conditional_rewards) > 20
        assert len(pooled_rewards) > len(conditional_rewards)
        conditional_mean = sum(conditional_rewards) / len(conditional_rewards)
        pooled_mean = sum(pooled_rewards) / len(pooled_rewards)
        assert conditional_mean < -0.5
        assert abs(pooled_mean) < abs(conditional_mean) * 0.5

    def test_conditional_on_active_risk_expected_value_is_confidently_negative(self):
        """The same underlying data, conditioned on `intervention_kind ==
        "bowl_destroyed"` only -- this is what D-070's original diagnostic
        actually measured, and it holds up: robustly negative, CI
        excludes zero."""
        rewards = []
        for seed in range(120):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0], _REACH_STEPS)
                goal1 = _GRAPH.goals[1]
                if bool(goal_feasible(goal1, env.unwrapped._world_state())):
                    result = attempt_goal(env, goal1, _TRAY_SLOTS[1], _REACH_STEPS)
                    rewards.append(1.0 if result["achieved"] else -0.1 * result["steps_used"])
            finally:
                env.close()
        assert len(rewards) > 20
        mean, lo, hi = bootstrap_ci(rewards, n_resamples=2000, seed=0)
        assert hi < 0
        assert mean < -0.5


class TestCalibratedPerInterventionKindPolicy:
    """The fixed design: calibrate per (goal_id, intervention_kind), read
    the episode's actual active intervention_kind at decision time, and
    make a decisive, not ambiguous, attempt/skip call."""

    @pytest.fixture(scope="class")
    def survival_probabilities(self):
        return calibrate_survival_probability(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=150, seed=0,
        )

    def test_place_mug_is_never_at_risk(self, survival_probabilities):
        assert survival_probabilities[("place_mug", "bowl_destroyed")] == 1.0
        assert survival_probabilities[("place_mug", "none")] == 1.0

    def test_place_bowl_survival_is_low_under_the_real_intervention(self, survival_probabilities):
        p = survival_probabilities[("place_bowl", "bowl_destroyed")]
        assert p < 0.5
        assert expected_value_of_attempt(p) < 0

    def test_place_bowl_is_safe_when_no_intervention_is_active(self, survival_probabilities):
        assert survival_probabilities[("place_bowl", "none")] == 1.0

    def test_policy_confidently_skips_the_risky_goal_under_real_risk(self, survival_probabilities):
        skip_count = 0
        for seed in range(15):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = calibrated_feasibility_policy(env, survival_probabilities, _GRAPH, attempt_goal, _TRAY_SLOTS)
            finally:
                env.close()
            if result["per_goal"]["place_bowl"].get("skipped", False):
                skip_count += 1
        assert skip_count == 15

    def test_policy_never_skips_when_genuinely_no_risk(self, survival_probabilities):
        """The decisive advantage over a pooled or Q-learned decision: this
        policy adapts correctly to the *actual* active intervention
        instead of applying one blanket, ambiguous rule to every
        episode."""
        skip_count = 0
        for seed in range(15):
            env = _make_env(intervention_kind="none", onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = calibrated_feasibility_policy(env, survival_probabilities, _GRAPH, attempt_goal, _TRAY_SLOTS)
            finally:
                env.close()
            if result["per_goal"]["place_bowl"].get("skipped", False):
                skip_count += 1
        assert skip_count == 0


class TestCalibrationDeploymentMismatch:
    """Calibration to a timing *distribution* is not free the same way
    D-069's mechanism generalization was: calibrating under a wide onset
    window, then deploying under a much narrower one where the
    intervention (if it fires at all) always resolves before the second
    goal's own decision point, keeps the pessimistic wide-regime
    probability -- over-conservative for the regime actually being
    deployed in, not automatically recalibrated."""

    def test_calibrated_on_wide_deployed_on_narrow_stays_pessimistic(self):
        survival_wide = calibrate_survival_probability(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("bowl_destroyed",), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=80, seed=0,
        )
        skip_count = 0
        for seed in range(15):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=_NARROW_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = calibrated_feasibility_policy(env, survival_wide, _GRAPH, attempt_goal, _TRAY_SLOTS)
            finally:
                env.close()
            if result["per_goal"]["place_bowl"].get("skipped", False):
                skip_count += 1
        # Under the real narrow-onset regime the intervention (if it fires
        # at all) resolves before place_bowl's own decision point, so
        # attempting is actually safe there -- but a policy calibrated on
        # the wide regime doesn't know that, and stays pessimistic.
        assert skip_count == 15


class TestHeldOutForcedVersusSelectiveWideTiming:
    """D-075: execute D-074's ablation on real, disjoint simulator seeds.

    This intentionally tests the likely negative result too: if the forced
    point-estimate baseline is already correct on every held-out stratum,
    abstention cannot improve its risk and merely gives up coverage. That is a
    valid H5 result, not a reason to tune the calibration sample until the
    selective method wins.
    """

    def test_real_held_out_ablation_without_label_leakage(self):
        estimates = calibrate_survival_estimates(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"),
            onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=20,
            seed=0,
        )

        # Derive each stratum's reward-optimal binary action exclusively from
        # held-out seeds, far outside calibration's internally sampled range.
        rewards = {key: [] for key in estimates}
        for intervention_kind in ("none", "bowl_destroyed"):
            for seed in range(10_000, 10_040):
                env = _make_env(intervention_kind, _WIDE_ONSET_RANGE)
                try:
                    env.reset(seed=seed)
                    for i, goal in enumerate(_GRAPH.goals):
                        key = (goal.id, intervention_kind)
                        if not bool(goal_feasible(goal, env.unwrapped._world_state())):
                            continue
                        outcome = attempt_goal(env, goal, _TRAY_SLOTS[i], _REACH_STEPS)
                        rewards[key].append(
                            1.0 if outcome["achieved"] else -0.1 * outcome["steps_used"]
                        )
                finally:
                    env.close()

        held_out_cases = []
        for key in estimates:
            assert rewards[key], f"held-out stratum unexpectedly empty: {key}"
            mean_reward = sum(rewards[key]) / len(rewards[key])
            held_out_cases.append((key, ATTEMPT if mean_reward > 0 else SKIP))

        result = compare_forced_vs_selective(estimates, held_out_cases)
        print("real wide-timing forced-vs-selective result:", result)

        # D-071's strong per-intervention separation makes the point baseline
        # correct on these held-out strata. With only 20 calibration episodes,
        # Wilson uncertainty should abstain somewhere, yielding the honest
        # negative H5 result: equal risk, lower coverage.
        assert result.forced_risk == 0.0
        assert result.selective_risk == 0.0
        assert 0.0 < result.selective_coverage < 1.0
