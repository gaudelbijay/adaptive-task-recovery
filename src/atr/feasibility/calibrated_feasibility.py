"""H5 (calibration): D-070 found that a binary feasibility check
(`goal_feasible()`) can't distinguish "safe" from "feasible now but at
risk of becoming infeasible mid-attempt" once intervention timing is wide
enough to span an attempt's own duration.

D-071 corrected D-070's specific causal claim (see `ai-notes/decisions.md`):
this module's first version calibrated a single survival probability per
goal, *pooled* across every `intervention_kind` the env can sample --
exactly the same pooling `train_q_table()`'s `(goal_id, feasible)` state
key does. Measured directly (D-071): that pooled true expected value is
statistically indistinguishable from zero (bootstrap 95% CI straddling
zero, `n=441`), because risk-free `"none"` episodes and genuinely risky
`"bowl_destroyed"` episodes get averaged into one number. What *is*
robustly, confidently negative (CI clearly excludes zero, `n=198`) is the
expected value *conditional on the risky intervention actually being
active*. Calibrating per `(goal_id, intervention_kind)` instead of pooling
recovers that decisive conditional signal directly -- `env.unwrapped.
intervention_kind` is privileged state, at the same privilege level
`goal_feasible()` itself already uses throughout this project (see
`atr.envs.tidy_up_env`), not a new kind of cheating.

`calibrate_survival_probability()` runs real rollouts and empirically
measures, per `(goal_id, intervention_kind)`, P(still achievable through
completion of its own attempt | perceived feasible right now);
`calibrated_feasibility_policy()` reads the episode's actual
`intervention_kind`, looks up the matching calibrated probability, and
attempts a perceived-feasible goal only when doing so has positive
expected value under the same reward shape `train_q_table()`
(`atr.policies.q_learning`) uses (`+1.0` achieved, `-0.1 * steps_used`
otherwise) -- a strict generalization of `feasibility_aware_policy`'s
binary rule (`atr.envs.tidy_up_policies`): it collapses back to "attempt
iff feasible" whenever the calibrated survival probability is high
enough, and grows more conservative as it drops.

Deliberately not reusing Q-learning's own machinery: the point is an
explicit, interpretable probability, calibrated once via direct
Monte-Carlo rollouts (and reported with a bootstrap CI via
`atr.evaluation.harness.bootstrap_ci`, D-042) rather than a small-sample,
recency-biased TD-learned table -- and, unlike a Q-table, it makes
visible whether it was calibrated under a timing distribution that
matches deployment (see the calibration/deployment mismatch test in
`tests/drafts/test_calibrated_feasibility.py`). That is a genuinely
different generalization axis from D-069's held-out-intervention-
*mechanism* result: D-069's generalization was close to free by
construction (the `(goal_id, feasible)` state never encoded mechanism, so
nothing could depend on it); calibration to a timing *distribution* is
not free the same way, and the mismatch test demonstrates that directly
rather than assuming it.

D-073 preserves the calibration counts instead of immediately collapsing
them to a point estimate.  A Wilson interval then supports a three-way
selective decision: attempt only when the whole interval has positive
expected value, skip only when the whole interval has negative expected
value, and abstain while the evidence still crosses the decision boundary.
The original point-probability API remains available for compatibility.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import sqrt
from typing import Callable

from atr.feasibility.oracle import goal_feasible
from atr.language.goal_graph import GoalGraph
from atr.policies.baselines import _summarize
from atr.policies.q_learning import _REACH_STEPS, _wait

_SUCCESS_REWARD = 1.0
_STEP_COST = 0.1

ATTEMPT = "attempt"
SKIP = "skip"
ABSTAIN = "abstain"


@dataclass(frozen=True)
class SurvivalEstimate:
    """Finite-sample survival estimate with a Wilson score interval.

    D-071 only retained the point probability, which cannot distinguish a
    well-supported 8/10 estimate from a single 1/1 observation.  Keeping the
    counts makes uncertainty explicit without pretending a binary Monte Carlo
    outcome is more precise than the evidence supports.
    """

    successes: int
    trials: int
    confidence: float = 0.95

    def __post_init__(self):
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not 0 <= self.successes <= self.trials:
            raise ValueError("successes must be between 0 and trials")
        if self.confidence != 0.95:
            raise ValueError("only the predeclared 95% interval is supported")

    @property
    def probability(self) -> float:
        return self.successes / self.trials

    @property
    def interval(self) -> tuple[float, float]:
        """95% Wilson score interval for a Bernoulli survival rate."""
        z = 1.959963984540054
        n = self.trials
        p = self.probability
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)


@dataclass(frozen=True)
class SelectiveAblationResult:
    """Held-out forced-classification versus abstention comparison."""

    forced_risk: float
    selective_risk: float
    selective_coverage: float
    forced_decisions: tuple[str, ...]
    selective_decisions: tuple[str, ...]


def selective_action(
    estimate: SurvivalEstimate,
    reach_steps: int = _REACH_STEPS,
) -> str:
    """Return ATTEMPT, SKIP, or ABSTAIN from the full confidence interval.

    Attempt only when even the interval's pessimistic endpoint has positive
    expected value; skip only when even its optimistic endpoint is negative.
    If the reward-optimal boundary lies inside the interval, evidence is
    genuinely ambiguous and the policy abstains instead of forcing a binary
    decision.  This is H5's first operational selective-prediction rule.
    """
    lo, hi = estimate.interval
    if expected_value_of_attempt(lo, reach_steps) > 0:
        return ATTEMPT
    if expected_value_of_attempt(hi, reach_steps) < 0:
        return SKIP
    return ABSTAIN


def selective_risk_coverage(
    decisions: list[str], correct_binary_actions: list[str]
) -> tuple[float, float]:
    """Return (selective risk, coverage) for attempt/skip predictions.

    Abstentions lower coverage and are excluded from selective risk.  Empty
    coverage has zero measured risk rather than dividing by zero; callers must
    always report both values so abstaining everywhere cannot look successful.
    """
    if len(decisions) != len(correct_binary_actions):
        raise ValueError("decisions and correct_binary_actions must have equal length")
    if any(action not in (ATTEMPT, SKIP, ABSTAIN) for action in decisions):
        raise ValueError("unknown selective decision")
    if any(action not in (ATTEMPT, SKIP) for action in correct_binary_actions):
        raise ValueError("correct actions must be attempt or skip")
    answered = [i for i, action in enumerate(decisions) if action != ABSTAIN]
    coverage = len(answered) / len(decisions) if decisions else 0.0
    risk = (
        sum(decisions[i] != correct_binary_actions[i] for i in answered) / len(answered)
        if answered else 0.0
    )
    return risk, coverage


def compare_forced_vs_selective(
    calibration_estimates: dict[tuple[str, str], SurvivalEstimate],
    held_out_cases: list[tuple[tuple[str, str], str]],
    reach_steps: int = _REACH_STEPS,
) -> SelectiveAblationResult:
    """Evaluate D-073's predeclared ablation on held-out binary labels.

    Each held-out case is ``((goal_id, intervention_kind), correct_action)``.
    Estimates must have been fitted before these labels were observed.  The
    forced baseline thresholds the point probability; the selective method
    uses the same estimate and reward boundary but may abstain based on its
    interval.  Missing calibration is an abstention for the selective method
    and an error: the forced baseline cannot manufacture a binary prediction
    without evidence, so missing keys fail loudly instead of receiving a
    favorable default.
    """
    forced: list[str] = []
    selective: list[str] = []
    correct: list[str] = []
    for key, correct_action in held_out_cases:
        if correct_action not in (ATTEMPT, SKIP):
            raise ValueError("held-out correct actions must be attempt or skip")
        if key not in calibration_estimates:
            raise ValueError(f"no calibration estimate for held-out key {key!r}")
        estimate = calibration_estimates[key]
        forced.append(
            ATTEMPT
            if expected_value_of_attempt(estimate.probability, reach_steps) > 0
            else SKIP
        )
        selective.append(selective_action(estimate, reach_steps))
        correct.append(correct_action)

    forced_risk, forced_coverage = selective_risk_coverage(forced, correct)
    selective_risk, selective_coverage = selective_risk_coverage(selective, correct)
    if forced_coverage != (1.0 if held_out_cases else 0.0):
        raise AssertionError("forced baseline must answer every held-out case")
    return SelectiveAblationResult(
        forced_risk=forced_risk,
        selective_risk=selective_risk,
        selective_coverage=selective_coverage,
        forced_decisions=tuple(forced),
        selective_decisions=tuple(selective),
    )


@dataclass(frozen=True)
class RewardComparisonResult:
    """D-077: whether selective abstention's coverage cost is actually
    "worth it" in the project's own reward terms, not just risk/coverage
    counts. `forced_risk`/`selective_risk` (`compare_forced_vs_selective()`)
    treat every wrong decision as equally bad and every abstention as free
    -- neither is true here: a wrong ATTEMPT on a stratum with true
    survival probability 0.6 costs less in expectation than one on a
    stratum with true survival 0.05, and abstaining isn't free, it's a
    small, explicit wait cost. This answers the question in the reward
    units already used everywhere else in this project."""

    forced_mean_reward: float
    selective_mean_reward: float
    forced_rewards: tuple[float, ...]
    selective_rewards: tuple[float, ...]


def expected_reward_of_decision(
    decision: str,
    true_survival_probability: float,
    reach_steps: int = _REACH_STEPS,
    abstain_steps: int = 1,
) -> float:
    """Real expected reward of a single ATTEMPT/SKIP/ABSTAIN decision, given
    a stratum's *true* survival probability. Extends the same reward shape
    `train_q_table()` and `expected_value_of_attempt()` already use
    (`+1.0` achieved, `-0.1 * steps_used` otherwise) to the ABSTAIN action --
    a small, explicit `-0.1 * abstain_steps` wait cost, matching
    `selective_calibrated_policy()`'s own `abstain_steps` semantics -- rather
    than inventing a new cost function. SKIP's reward is always 0.0, same as
    every other policy in this project (`train_q_table()`'s SKIP branch,
    `feasibility_aware_policy`'s skip case): no goal credit, no steps
    wasted."""
    if decision == ATTEMPT:
        return expected_value_of_attempt(true_survival_probability, reach_steps)
    if decision == SKIP:
        return 0.0
    if decision == ABSTAIN:
        return -_STEP_COST * abstain_steps
    raise ValueError(f"unknown decision {decision!r}")


def compare_forced_vs_selective_reward(
    calibration_estimates: dict[tuple[str, str], SurvivalEstimate],
    held_out_true_probabilities: dict[tuple[str, str], float],
    reach_steps: int = _REACH_STEPS,
    abstain_steps: int = 1,
) -> RewardComparisonResult:
    """Reward-unit counterpart to `compare_forced_vs_selective()`. Each
    held-out stratum's *true* survival probability (not just a binary
    correct-action label) lets `expected_reward_of_decision()` score a wrong
    ATTEMPT by how wrong it actually was, and score abstention by its real,
    small cost rather than folding it into "coverage" as a separate axis.
    `held_out_true_probabilities` should come from a held-out sample
    disjoint from whatever produced `calibration_estimates`, same
    requirement `compare_forced_vs_selective()` has."""
    forced_rewards: list[float] = []
    selective_rewards: list[float] = []
    for key, true_p in held_out_true_probabilities.items():
        if key not in calibration_estimates:
            raise ValueError(f"no calibration estimate for held-out key {key!r}")
        estimate = calibration_estimates[key]
        forced_decision = (
            ATTEMPT if expected_value_of_attempt(estimate.probability, reach_steps) > 0 else SKIP
        )
        selective_decision = selective_action(estimate, reach_steps)
        forced_rewards.append(expected_reward_of_decision(forced_decision, true_p, reach_steps, abstain_steps))
        selective_rewards.append(
            expected_reward_of_decision(selective_decision, true_p, reach_steps, abstain_steps)
        )
    return RewardComparisonResult(
        forced_mean_reward=sum(forced_rewards) / len(forced_rewards),
        selective_mean_reward=sum(selective_rewards) / len(selective_rewards),
        forced_rewards=tuple(forced_rewards),
        selective_rewards=tuple(selective_rewards),
    )


def calibrate_survival_probability(
    make_env: Callable,
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, str] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 150,
    seed: int = 0,
) -> dict[tuple[str, str], float]:
    """Empirically measures, per `(goal_id, intervention_kind)`, P(actually
    achieved | perceived feasible at this goal's own decision point) via
    real rollouts that always attempt when perceived feasible -- so the
    measurement isn't contaminated by any policy's own skip decisions.
    Keyed by `intervention_kind`, not pooled across it (D-071 -- see
    module docstring for why pooling produces a statistically ambiguous
    number instead of a decisive one). A `(goal_id, intervention_kind)`
    with zero perceived-feasible-then-attempted observations gets 1.0 (no
    evidence of risk -- defaults to the naive "attempt iff feasible"
    assumption rather than an arbitrary pessimistic one)."""
    feasible_counts, achieved_counts = _collect_survival_counts(
        make_env, graph, tray_slots, attempt_goal_fn, intervention_kinds,
        onset_step_bounds, reach_steps, n_episodes, seed,
    )
    return {key: achieved_counts.get(key, 0) / count for key, count in feasible_counts.items()}


def _collect_survival_counts(
    make_env: Callable,
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, ...],
    onset_step_bounds: tuple[int, int],
    reach_steps: int,
    n_episodes: int,
    seed: int,
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    rng = random.Random(seed)
    feasible_counts: dict[tuple[str, str], int] = {}
    achieved_counts: dict[tuple[str, str], int] = {}
    for _ in range(n_episodes):
        intervention_kind = rng.choice(intervention_kinds)
        onset_step = rng.randint(*onset_step_bounds)
        env = make_env(intervention_kind, (onset_step, onset_step + 1))
        try:
            env.reset(seed=rng.randint(0, 2**31 - 1))
            for i, goal in enumerate(graph.goals):
                if not bool(goal_feasible(goal, env.unwrapped._world_state())):
                    _wait(env, reach_steps)
                    continue
                key = (goal.id, intervention_kind)
                feasible_counts[key] = feasible_counts.get(key, 0) + 1
                result = attempt_goal_fn(env, goal, tray_slots[i], reach_steps)
                if result["achieved"]:
                    achieved_counts[key] = achieved_counts.get(key, 0) + 1
        finally:
            env.close()
    return feasible_counts, achieved_counts


def calibrate_survival_estimates(
    make_env: Callable,
    graph: GoalGraph,
    tray_slots: list,
    attempt_goal_fn: Callable,
    intervention_kinds: tuple[str, str] = ("none", "bowl_destroyed"),
    onset_step_bounds: tuple[int, int] = (1, 4),
    reach_steps: int = _REACH_STEPS,
    n_episodes: int = 150,
    seed: int = 0,
) -> dict[tuple[str, str], SurvivalEstimate]:
    """Count-preserving counterpart to `calibrate_survival_probability()`.

    It uses the same rollout-counting path as the point API rather than trying
    to reconstruct sample sizes from probabilities. The old API remains
    behavior-compatible; new selective callers opt into this richer one.
    """
    feasible_counts, achieved_counts = _collect_survival_counts(
        make_env, graph, tray_slots, attempt_goal_fn, intervention_kinds,
        onset_step_bounds, reach_steps, n_episodes, seed,
    )
    return {
        key: SurvivalEstimate(achieved_counts.get(key, 0), count)
        for key, count in feasible_counts.items()
    }


def expected_value_of_attempt(survival_probability: float, reach_steps: int = _REACH_STEPS) -> float:
    """Expected reward of attempting a perceived-feasible goal, matching
    `train_q_table()`'s exact reward shape so a calibrated policy's
    decision threshold (attempt iff this is positive) is directly
    comparable to what a Q-learned policy converges to, not a separately
    invented one."""
    p = survival_probability
    return p * _SUCCESS_REWARD - (1 - p) * _STEP_COST * reach_steps


def calibrated_feasibility_policy(
    env,
    survival_probabilities: dict[tuple[str, str], float],
    graph: GoalGraph,
    attempt_goal_fn: Callable,
    tray_slots: list,
) -> dict:
    """Same result shape as static_policy/feasibility_aware_policy/
    learned_policy for direct comparison. Reads the episode's actual
    `intervention_kind` (privileged state, same level `goal_feasible()`
    already uses) to select the matching calibrated probability, and
    attempts a perceived-feasible goal only when
    `expected_value_of_attempt()` on that probability is positive."""
    intervention_kind = env.unwrapped.intervention_kind
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        p = survival_probabilities.get((goal.id, intervention_kind), 1.0)
        if feasible and expected_value_of_attempt(p) > 0:
            per_goal[goal.id] = attempt_goal_fn(env, goal, tray_slots[i])
        else:
            per_goal[goal.id] = {"achieved": False, "steps_used": 0, "skipped": True}
    return _summarize(per_goal)


def selective_calibrated_policy(
    env,
    survival_estimates: dict[tuple[str, str], SurvivalEstimate],
    graph: GoalGraph,
    attempt_goal_fn: Callable,
    tray_slots: list,
    abstain_steps: int = 1,
) -> dict:
    """Uncertainty-aware counterpart to `calibrated_feasibility_policy()`.

    A missing estimate is treated as uncertain, not silently optimistic.
    Abstention incurs a small, explicit wait cost and is recorded separately
    from an intentional skip so evaluation cannot mistake indecision for a
    confident strategy choice.
    """
    intervention_kind = env.unwrapped.intervention_kind
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        feasible = bool(goal_feasible(goal, env.unwrapped._world_state()))
        estimate = survival_estimates.get((goal.id, intervention_kind))
        decision = selective_action(estimate) if feasible and estimate is not None else (
            ABSTAIN if feasible else SKIP
        )
        if decision == ATTEMPT:
            outcome = attempt_goal_fn(env, goal, tray_slots[i])
        elif decision == ABSTAIN:
            _wait(env, abstain_steps)
            outcome = {
                "achieved": False,
                "steps_used": abstain_steps,
                "skipped": False,
                "abstained": True,
            }
        else:
            outcome = {"achieved": False, "steps_used": 0, "skipped": True}
        outcome["decision"] = decision
        per_goal[goal.id] = outcome
    result = _summarize(per_goal)
    result["abstentions"] = sum(
        outcome.get("abstained", False) for outcome in per_goal.values()
    )
    result["decision_coverage"] = (
        (len(per_goal) - result["abstentions"]) / len(per_goal) if per_goal else 0.0
    )
    return result
