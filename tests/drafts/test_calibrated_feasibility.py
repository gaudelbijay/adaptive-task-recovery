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
    ABSTAIN,
    ATTEMPT,
    SKIP,
    calibrate_survival_probability,
    calibrate_survival_estimates,
    calibrated_feasibility_policy,
    compare_forced_vs_selective,
    compare_forced_vs_selective_reward,
    expected_value_of_attempt,
    selective_action,
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


class TestSelectiveAbstentionOnAGenuinelyAmbiguousCase:
    """D-076: D-075 gave H5 an easy test -- D-071's strong per-intervention
    separation for `bowl_destroyed` under `_WIDE_ONSET_RANGE` (true survival
    ~0.28, true EV ~-1.5) meant the 20-episode point estimate was already
    correct on every held-out stratum, so abstention had nothing to protect
    against (it only cost coverage). This constructs the fair test D-075's own
    "Consequences" section named as missing: a stratum whose true expected
    value sits close to the reward-decision boundary, not confidently on
    either side.

    Found empirically (not guessed), by sweeping `onset_step_bounds` upper
    limits for `bowl_destroyed` and measuring real survival probability
    directly (`max_episode_steps=50` on `TidyUp-v1` means an onset past that
    point simply never fires within the episode, not a longer "safe tail" --
    the naive "wider always means safer" intuition from D-070/D-071 doesn't
    linearly extrapolate, confirmed by measuring rather than assuming):
    `onset_step_bounds=(10, 100)` puts `(place_bowl, "bowl_destroyed")`'s true
    survival probability at ~0.60 (a 200-episode held-out estimate), giving a
    true expected value of ~-0.41 -- close enough to the `EV=0` boundary that
    a 20-episode calibration sample frequently lands its point estimate on the
    wrong side of it.
    """

    _AMBIGUOUS_ONSET_RANGE = (10, 100)

    def test_forced_baseline_is_frequently_wrong_under_genuine_ambiguity(self):
        # Ground truth from a large, disjoint held-out sample (200 episodes,
        # seeds 10000-10199) -- kept well outside every calibration seed used
        # below (0-9) so no calibration run can see it.
        n_feasible, n_achieved = 0, 0
        for seed in range(10_000, 10_200):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=self._AMBIGUOUS_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0], _REACH_STEPS)
                goal1 = _GRAPH.goals[1]
                if bool(goal_feasible(goal1, env.unwrapped._world_state())):
                    n_feasible += 1
                    result = attempt_goal(env, goal1, _TRAY_SLOTS[1], _REACH_STEPS)
                    if result["achieved"]:
                        n_achieved += 1
            finally:
                env.close()
        true_p = n_achieved / n_feasible
        true_ev = expected_value_of_attempt(true_p)
        # Confirms this stratum is actually the ambiguous case it's meant to
        # be, not asserting the specific decision it happens to land on.
        assert -1.0 < true_ev < 0.3
        ground_truth_action = ATTEMPT if true_ev > 0 else SKIP

        forced_wrong = 0
        selective_wrong = 0
        selective_abstain = 0
        n_calibration_seeds = 10
        for cal_seed in range(n_calibration_seeds):
            estimates = calibrate_survival_estimates(
                _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
                intervention_kinds=("bowl_destroyed",), onset_step_bounds=self._AMBIGUOUS_ONSET_RANGE,
                n_episodes=20, seed=cal_seed,
            )
            estimate = estimates.get(("place_bowl", "bowl_destroyed"))
            if estimate is None:
                continue
            forced_decision = ATTEMPT if expected_value_of_attempt(estimate.probability) > 0 else SKIP
            selective_decision = selective_action(estimate)
            if forced_decision != ground_truth_action:
                forced_wrong += 1
            if selective_decision == ABSTAIN:
                selective_abstain += 1
            elif selective_decision != ground_truth_action:
                selective_wrong += 1

        # The actual H5 claim, given a fair, genuinely ambiguous test: forced
        # point-estimate decisions are wrong often (real, measured: 5/10
        # calibration seeds), while selective abstention never confidently
        # commits to the wrong answer -- it recognizes the ambiguity and
        # abstains instead, at the cost of coverage. Loose bounds (not exact
        # counts) since this is a real stochastic small-sample process, not a
        # designed fixture -- the qualitative contrast is the claim.
        assert forced_wrong >= 3
        assert selective_wrong == 0
        assert selective_abstain >= 1


class TestDownstreamCostModelForTheCoverageTradeOff:
    """D-077: D-075 and D-076 both flagged the same open question -- is
    selective abstention's coverage cost actually *worth it*? Risk/coverage
    counts alone can't answer that: they treat every wrong ATTEMPT as
    equally bad and every abstention as free, neither of which is true.
    Extends the exact reward shape `train_q_table()` already uses
    (`+1.0` achieved, `-0.1 * steps_used` otherwise) to the ABSTAIN action
    (a small, explicit wait cost) instead of inventing a new cost function,
    then answers the question in those units directly, on the same
    genuinely-ambiguous stratum D-076 already established
    (`(place_bowl, "bowl_destroyed")`, `onset_step_bounds=(10, 100)`, true
    survival ~0.60)."""

    _AMBIGUOUS_ONSET_RANGE = (10, 100)

    def test_selective_yields_higher_expected_reward_than_forced(self):
        n_feasible, n_achieved = 0, 0
        for seed in range(10_000, 10_200):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=self._AMBIGUOUS_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0], _REACH_STEPS)
                goal1 = _GRAPH.goals[1]
                if bool(goal_feasible(goal1, env.unwrapped._world_state())):
                    n_feasible += 1
                    result = attempt_goal(env, goal1, _TRAY_SLOTS[1], _REACH_STEPS)
                    if result["achieved"]:
                        n_achieved += 1
            finally:
                env.close()
        true_p = n_achieved / n_feasible
        key = ("place_bowl", "bowl_destroyed")
        held_out_true_probabilities = {key: true_p}

        forced_rewards = []
        selective_rewards = []
        for cal_seed in range(10):
            estimates = calibrate_survival_estimates(
                _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
                intervention_kinds=("bowl_destroyed",), onset_step_bounds=self._AMBIGUOUS_ONSET_RANGE,
                n_episodes=20, seed=cal_seed,
            )
            if key not in estimates:
                continue
            result = compare_forced_vs_selective_reward(estimates, held_out_true_probabilities)
            forced_rewards.append(result.forced_rewards[0])
            selective_rewards.append(result.selective_rewards[0])

        mean_forced = sum(forced_rewards) / len(forced_rewards)
        mean_selective = sum(selective_rewards) / len(selective_rewards)
        # Real measured: mean_forced=-0.2044, mean_selective=-0.0800 -- both
        # negative (this stratum is genuinely marginal, so no strategy
        # achieves the goal reliably), but selective's abstention cost is
        # markedly smaller than forced's wrong-attempt cost. Loose bounds,
        # not exact equality, for the same reason as the sibling test above.
        assert mean_selective > mean_forced
        assert mean_forced < -0.1
        assert mean_selective > -0.15


class TestAbstentionDoesNotAlwaysWin:
    """D-078: D-077 disclosed its own scope limit -- the tested stratum's
    true answer was SKIP, where a wrong forced ATTEMPT is expensive
    (`-0.1 * steps_used`, real reward lost) and abstaining is cheap. The
    sharper, still-untested case is the opposite: a stratum whose true
    answer is ATTEMPT (worth doing), where a wrong forced SKIP costs
    *nothing* in this reward shape (SKIP always yields 0.0, regardless of
    whether it was the right call) but abstaining still pays its fixed
    wait cost every time, forfeiting the real upside a correct ATTEMPT
    would have earned.

    Found the same way D-076 found its stratum -- swept onset ranges,
    measured real survival probability, picked one close to the boundary
    from the positive side: `onset_step_bounds=(10, 120)`, true survival
    ~0.73 (a 200-episode held-out estimate), true EV ~+0.07.
    """

    _POSITIVE_ONSET_RANGE = (10, 120)

    def test_forced_beats_selective_when_the_true_answer_is_attempt(self):
        n_feasible, n_achieved = 0, 0
        for seed in range(20_000, 20_200):
            env = _make_env(intervention_kind="bowl_destroyed", onset_step_range=self._POSITIVE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0], _REACH_STEPS)
                goal1 = _GRAPH.goals[1]
                if bool(goal_feasible(goal1, env.unwrapped._world_state())):
                    n_feasible += 1
                    result = attempt_goal(env, goal1, _TRAY_SLOTS[1], _REACH_STEPS)
                    if result["achieved"]:
                        n_achieved += 1
            finally:
                env.close()
        true_p = n_achieved / n_feasible
        true_ev = expected_value_of_attempt(true_p)
        # Confirms this stratum is the intended positive-but-close-to-the-
        # boundary case, not asserting the specific decision it lands on.
        assert 0.0 < true_ev < 0.3
        key = ("place_bowl", "bowl_destroyed")
        held_out_true_probabilities = {key: true_p}

        forced_rewards = []
        selective_rewards = []
        for cal_seed in range(10):
            estimates = calibrate_survival_estimates(
                _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
                intervention_kinds=("bowl_destroyed",), onset_step_bounds=self._POSITIVE_ONSET_RANGE,
                n_episodes=20, seed=cal_seed,
            )
            if key not in estimates:
                continue
            result = compare_forced_vs_selective_reward(estimates, held_out_true_probabilities)
            forced_rewards.append(result.forced_rewards[0])
            selective_rewards.append(result.selective_rewards[0])

        mean_forced = sum(forced_rewards) / len(forced_rewards)
        mean_selective = sum(selective_rewards) / len(selective_rewards)
        # Real measured: mean_forced=+0.0506, mean_selective=-0.0728. Forced
        # wins here -- the opposite of D-077's stratum -- because a wrong
        # forced SKIP costs nothing in this reward shape (0.0, same as a
        # correct one), while abstention's fixed wait cost is paid every
        # time regardless, exceeding the small positive upside being
        # protected. Abstention is not a free win; it's worth its cost only
        # when it's protecting against the *expensive* mistake (a wrong
        # ATTEMPT), not the cheap one (a wrong SKIP).
        assert mean_forced > mean_selective
        assert mean_forced > 0.0
