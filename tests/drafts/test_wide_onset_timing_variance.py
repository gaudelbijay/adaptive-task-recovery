"""D-070: gives the statistics machinery (bootstrap_ci/compare_policies,
D-042; track_comparison, D-057) genuine outcome variance to measure, and
locks in a real, substantive finding that surfaced while doing it.

Root cause of every zero-variance comparison in this project so far
(D-042's original H2 run, D-069's held-out-intervention run, and every
test file that passes `onset_step_range=(2, 3)`): `tidy_up_env.py`
samples onset timing via `rng.integers(*self.onset_step_range)` --
numpy's `Generator.integers()` is exclusive on the upper bound, so
`(2, 3)` always samples exactly 2. Not a bug (nothing in those tests
needed timing variance for what they were checking), but it meant the
statistical machinery has never had anything non-degenerate to report
on. A genuinely wide range (`(10, 60)`, spanning both `place_mug`'s own
~25-step attempt duration and `place_bowl`'s subsequent one) produces
real per-seed outcome variance -- confirmed directly (see
TestWideOnsetRangeProducesRealVariance).

The bigger finding, found investigating *why* the learned policy's
result looked different under wide timing, not assumed: with an onset
window wide enough to span both goals' attempt durations, "perceived
feasible right now" stops reliably predicting "will complete
successfully" -- measured directly (60 real episodes, always-attempt):
of the goals perceived feasible at their own decision point, 72.5%
(29/40) were destroyed mid-attempt anyway, because attempting itself
takes ~25 steps, comparable to the intervention's own timing spread.
Given this project's reward shape (+1.0 achieved, -0.1 * steps_used
otherwise), the expected value of attempting under that failure rate is
strongly negative (~-1.5), so a reward-trained Q-learning policy
correctly (not buggily) converges to skipping the goal even when
currently perceived feasible -- a genuinely different, and under this
specific reward shape actually reward-*optimal*, strategy from
`feasibility_aware_policy`'s hard-coded "attempt iff feasible" rule.
This is real evidence for exactly why docs/01/docs/10 insist on
reporting achieved-goals and wasted-steps *separately* rather than
collapsing them into one reward number: the reward-optimal policy here
looks *worse* on raw goal recall than the naive existence-check policy,
because the reward function encodes a specific risk tolerance the
existence check doesn't know about.
"""

import gymnasium as gym
import pytest

pytest.importorskip("mani_skill")

import task_schema_draft  # noqa: E402, F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal, feasibility_aware_policy  # noqa: E402
from atr.feasibility.oracle import goal_feasible  # noqa: E402
from atr.language.goal_graph import canonical_example  # noqa: E402
from atr.policies.q_learning import ATTEMPT, SKIP, learned_policy, train_q_table  # noqa: E402

_GRAPH = canonical_example()
_WIDE_ONSET_RANGE = (10, 60)


def _make_env(intervention_kind: str = "bowl_destroyed", onset_step_range=(5, 15)):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


class TestWideOnsetRangeProducesRealVariance:
    """The narrow default many tests use (e.g. `(2, 3)`) always samples
    the same onset step, by construction (numpy's exclusive upper
    bound) -- this confirms a genuinely wide range doesn't."""

    def test_onset_step_actually_varies_across_seeds(self):
        onset_steps_seen = set()
        for seed in range(15):
            env = _make_env(onset_step_range=_WIDE_ONSET_RANGE)
            env.reset(seed=seed)
            onset_steps_seen.add(env.unwrapped._onset_step)
            env.close()
        assert len(onset_steps_seen) > 1

    def test_goals_achieved_actually_varies_across_seeds(self):
        """The thing D-042 never got to observe: an outcome metric that
        isn't identical across every seed, given the same policy."""
        outcomes = set()
        for seed in range(15):
            env = _make_env(onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                result = feasibility_aware_policy(env)
                outcomes.add(result["goals_achieved"])
            finally:
                env.close()
        assert len(outcomes) > 1


class TestPerceivedFeasibleDoesNotReliablyPredictCompletion:
    """The real, measured, non-obvious finding: once the onset window is
    wide enough to span the *second* goal's own attempt duration, being
    perceived feasible at decision time is a poor predictor of actually
    completing -- confirmed directly by running the always-attempt
    behavior and comparing the feasibility read at decision time against
    the real outcome afterward, not inferred from Q-values alone."""

    def test_majority_of_perceived_feasible_cases_fail_mid_attempt(self):
        mismatches = 0
        total_perceived_feasible = 0
        for seed in range(40):
            env = _make_env(onset_step_range=_WIDE_ONSET_RANGE)
            try:
                env.reset(seed=seed)
                attempt_goal(env, _GRAPH.goals[0], _TRAY_SLOTS[0])
                perceived_feasible = bool(goal_feasible(_GRAPH.goals[1], env.unwrapped._world_state()))
                if perceived_feasible:
                    total_perceived_feasible += 1
                    result = attempt_goal(env, _GRAPH.goals[1], _TRAY_SLOTS[1])
                    if not result["achieved"]:
                        mismatches += 1
            finally:
                env.close()
        assert total_perceived_feasible > 0
        # Real measured rate was 72.5% (29/40) -- asserting "more than
        # half" rather than the exact figure, since this is a genuinely
        # stochastic quantity across a different seed range than the one
        # originally measured (ai-notes/decisions.md D-070).
        assert mismatches / total_perceived_feasible > 0.5


class TestLearnedPolicyCorrectlyBecomesMoreConservativeUnderWideTiming:
    """Not a Q-learning bug -- the reward-optimal response, given this
    project's reward shape (+1.0 achieved, -0.1 * steps_used otherwise)
    and the ~72.5% mid-attempt failure rate confirmed above: expected
    value of attempting a "currently feasible" goal is strongly negative
    under this training distribution, so a reward-trained policy
    correctly learns to skip it -- a genuinely different, and under this
    specific reward shape actually reward-optimal, strategy from
    feasibility_aware_policy's hard-coded "attempt iff feasible" rule."""

    def test_q_table_prefers_skip_even_when_perceived_feasible(self):
        q_table = train_q_table(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=150, seed=0,
        )
        # place_mug is always feasible and always worth attempting --
        # unaffected by this timing risk, since nothing threatens it.
        assert q_table[("place_mug", True)][ATTEMPT] > q_table[("place_mug", True)][SKIP]
        # place_bowl, even when perceived feasible, is not worth the risk
        # under this reward shape and training distribution.
        assert q_table[("place_bowl", True)][SKIP] > q_table[("place_bowl", True)][ATTEMPT]

    def test_learned_policy_wastes_zero_steps_but_also_never_achieves_the_risky_goal(self):
        """The real, honest trade-off this reveals -- not just "skips
        more," but *why* that's the reward-optimal call here: zero
        wasted steps, at the cost of also never realizing the ~18% of
        cases (11/60 in the original measurement) where attempting
        would have actually succeeded. feasibility_aware_policy takes
        the opposite trade: it captures that upside, at the cost of
        wasting steps in the majority of cases where it doesn't pan
        out. Neither is "wrong" -- they optimize different things,
        exactly why docs/01/docs/10 report goals-achieved and
        wasted-steps separately rather than collapsing them into one
        number."""
        q_table = train_q_table(
            _make_env, _GRAPH, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=("none", "bowl_destroyed"), onset_step_bounds=_WIDE_ONSET_RANGE,
            n_episodes=150, seed=0,
        )
        env = _make_env(onset_step_range=_WIDE_ONSET_RANGE)
        try:
            env.reset(seed=0)
            result = learned_policy(env, q_table, _GRAPH, attempt_goal, _TRAY_SLOTS)
        finally:
            env.close()
        assert result["wasted_steps"] == 0
        assert result["per_goal"]["place_bowl"]["skipped"] is True
