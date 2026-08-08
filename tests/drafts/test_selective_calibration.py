"""D-073: finite-sample uncertainty and abstention for H5.

Pure-function tests deliberately run without ManiSkill so the reliable CI tier
checks the decision boundary, interval behavior, and risk/coverage accounting.
Live rollout calibration remains covered by test_calibrated_feasibility.py.
"""

import pytest

from atr.feasibility.calibrated_feasibility import (
    ABSTAIN,
    ATTEMPT,
    SKIP,
    SelectiveAblationResult,
    SurvivalEstimate,
    compare_forced_vs_selective,
    selective_action,
    selective_calibrated_policy,
    selective_risk_coverage,
)
from atr.feasibility.oracle import ObjectState
from atr.language.goal_graph import Goal, GoalGraph


class _ActionSpace:
    shape = (1,)

    def sample(self):
        return 0


class _FakeEnv:
    def __init__(self, intervention_kind="risk"):
        self.unwrapped = self
        self.intervention_kind = intervention_kind
        self.action_space = _ActionSpace()
        self.waited = 0

    def _world_state(self):
        return {"object": ObjectState(exists=True, position=None, up_vector=None)}

    def step(self, action):
        self.waited += 1


class TestSurvivalEstimate:
    def test_probability_retains_the_observed_frequency(self):
        assert SurvivalEstimate(8, 10).probability == 0.8

    def test_more_evidence_narrows_uncertainty_at_the_same_probability(self):
        small = SurvivalEstimate(8, 10).interval
        large = SurvivalEstimate(80, 100).interval
        assert large[1] - large[0] < small[1] - small[0]

    def test_extreme_rates_do_not_claim_zero_uncertainty(self):
        lo, hi = SurvivalEstimate(1, 1).interval
        assert lo < 1.0
        assert hi == 1.0

    @pytest.mark.parametrize(
        "successes,trials",
        [(-1, 1), (2, 1), (0, 0)],
    )
    def test_invalid_counts_raise(self, successes, trials):
        with pytest.raises(ValueError):
            SurvivalEstimate(successes, trials)


class TestSelectiveAction:
    def test_attempts_only_when_full_interval_is_reward_positive(self):
        assert selective_action(SurvivalEstimate(99, 100)) == ATTEMPT

    def test_skips_only_when_full_interval_is_reward_negative(self):
        assert selective_action(SurvivalEstimate(5, 100)) == SKIP

    def test_abstains_when_reward_boundary_lies_inside_interval(self):
        # Point estimate 0.8 favors ATTEMPT at the project's reward boundary
        # (~0.714), but 8/10 is not enough evidence for the full interval to.
        assert selective_action(SurvivalEstimate(8, 10)) == ABSTAIN

    def test_same_point_estimate_becomes_decisive_with_more_evidence(self):
        assert selective_action(SurvivalEstimate(800, 1000)) == ATTEMPT


class TestSelectiveRiskCoverage:
    def test_reports_error_only_on_answered_cases_and_coverage_separately(self):
        risk, coverage = selective_risk_coverage(
            [ATTEMPT, ABSTAIN, SKIP, ATTEMPT],
            [ATTEMPT, SKIP, SKIP, SKIP],
        )
        assert risk == pytest.approx(1 / 3)
        assert coverage == 0.75

    def test_abstaining_everywhere_cannot_hide_zero_coverage(self):
        risk, coverage = selective_risk_coverage(
            [ABSTAIN, ABSTAIN], [ATTEMPT, SKIP]
        )
        assert risk == 0.0
        assert coverage == 0.0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError):
            selective_risk_coverage([ATTEMPT], [])


class TestHeldOutForcedVersusSelectiveAblation:
    """Calibration counts and evaluation labels are deliberately separate.

    The threshold-near training stratum is the failure case H5 targets: its
    noisy point estimate lands above the action boundary although its held-out
    correct action is SKIP. Selective prediction declines that one call while
    retaining the two well-supported calls.
    """

    estimates = {
        ("safe", "none"): SurvivalEstimate(99, 100),
        ("risky", "change"): SurvivalEstimate(5, 100),
        ("boundary", "change"): SurvivalEstimate(8, 10),
    }
    held_out = [
        (("safe", "none"), ATTEMPT),
        (("risky", "change"), SKIP),
        (("boundary", "change"), SKIP),
    ]

    def test_abstention_reduces_held_out_risk_at_disclosed_coverage(self):
        result = compare_forced_vs_selective(self.estimates, self.held_out)
        assert isinstance(result, SelectiveAblationResult)
        assert result.forced_risk == pytest.approx(1 / 3)
        assert result.selective_risk == 0.0
        assert result.selective_coverage == pytest.approx(2 / 3)
        assert result.forced_decisions == (ATTEMPT, SKIP, ATTEMPT)
        assert result.selective_decisions == (ATTEMPT, SKIP, ABSTAIN)

    def test_more_calibration_evidence_restores_full_coverage(self):
        estimates = dict(self.estimates)
        estimates[("boundary", "change")] = SurvivalEstimate(800, 1000)
        held_out = self.held_out[:-1] + [(("boundary", "change"), ATTEMPT)]
        result = compare_forced_vs_selective(estimates, held_out)
        assert result.selective_risk == 0.0
        assert result.selective_coverage == 1.0

    def test_evaluation_fails_loudly_without_training_evidence(self):
        with pytest.raises(ValueError, match="no calibration estimate"):
            compare_forced_vs_selective({}, [(('new', 'change'), SKIP)])


class TestSelectivePolicy:
    graph = GoalGraph(
        instruction_text="Put the object on the tray.",
        goals=(Goal(id="place_object", predicate="on_tray", target_object="object"),),
        constraints=(),
    )

    def test_ambiguous_evidence_records_a_costed_abstention(self):
        env = _FakeEnv()
        attempts = []

        def attempt(*args):
            attempts.append(args)
            return {"achieved": True, "steps_used": 1, "skipped": False}

        result = selective_calibrated_policy(
            env,
            {("place_object", "risk"): SurvivalEstimate(8, 10)},
            self.graph,
            attempt,
            [None],
            abstain_steps=2,
        )
        assert attempts == []
        assert env.waited == 2
        assert result["abstentions"] == 1
        assert result["decision_coverage"] == 0.0
        assert result["per_goal"]["place_object"]["decision"] == ABSTAIN

    def test_well_supported_evidence_executes_the_attempt(self):
        env = _FakeEnv()
        result = selective_calibrated_policy(
            env,
            {("place_object", "risk"): SurvivalEstimate(99, 100)},
            self.graph,
            lambda *args: {"achieved": True, "steps_used": 1, "skipped": False},
            [None],
        )
        assert result["goals_achieved"] == 1
        assert result["abstentions"] == 0
        assert result["decision_coverage"] == 1.0
