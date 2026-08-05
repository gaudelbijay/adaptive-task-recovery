"""Real implementation of docs/10-evaluation-and-benchmarks.md's
"Statistical protocol" (D-042): "Predeclare primary metrics and splits.
Use paired episode seeds across methods, bootstrap confidence intervals."

Env-agnostic and policy-agnostic on purpose, same reasoning as
policies/baselines.py (D-040) and policies/q_learning.py (D-041): takes
an `env_factory` and a `{name: policy_fn}` mapping, not anything
canonical-env-specific, so the same harness works for any TidyUp env
variant and any policy (static/feasibility-aware/naive-substitution/
learned) without modification.

What this is not: a full implementation of docs/10's required-baselines
list or ablation suite -- those need baselines that don't exist yet
(domain-randomized policy, frame-difference detector, symbolic
replanner). This is the statistical machinery underneath whichever
comparison gets run, built once so it's not reinvented ad hoc per
comparison the way `_summarize()` was reinvented four times before D-040.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np

from atr.evaluation.logging import append_episode_log, build_episode_log
from atr.language.goal_graph import GoalGraph

EnvFactory = Callable[[], "gym.Env"]
PolicyFn = Callable[..., dict]


def run_episode(
    env_factory: EnvFactory, policy_fn: PolicyFn, seed: int,
    *, graph: GoalGraph | None = None, log_path: str | None = None,
) -> dict:
    """One episode: fresh env, reset at `seed`, run `policy_fn`, close.
    `policy_fn(env) -> dict` matching static_policy/feasibility_aware_policy/
    naive_substitution_policy/learned_policy's existing result shape
    (must include whatever metric keys the caller asks bootstrap_ci for).

    `log_path` (optional, off by default -- no behavior change for any
    existing caller): if given, also builds and appends a structured
    episode log (`atr.evaluation.logging`, the docs/03 step-6 "log
    interface"). Requires `graph`, since resolving each goal id to the
    object it targets -- needed to attach the oracle label -- isn't
    possible from the policy's result dict alone."""
    env = env_factory()
    try:
        env.reset(seed=seed)
        result = policy_fn(env)
        if log_path is not None:
            if graph is None:
                raise ValueError("log_path requires graph (to resolve goal -> target_object)")
            oracle_exists = dict(env.unwrapped._exists)
            append_episode_log(log_path, build_episode_log(result, graph, oracle_exists, seed=seed))
        return result
    finally:
        env.close()


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, ci: float = 0.95, seed: int = 0,
) -> tuple[float, float, float]:
    """(mean, lo, hi) -- percentile bootstrap confidence interval over
    `values` (one value per episode/seed). The standard nonparametric
    choice when nothing is known about the underlying distribution, which
    is exactly this project's situation (metrics like `goals_achieved`
    aren't Gaussian, especially at small sample counts)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resample_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(resample_means, [alpha, 1 - alpha])
    return float(values.mean()), float(lo), float(hi)


def compare_policies(
    env_factory: EnvFactory,
    policies: dict[str, PolicyFn],
    seeds: list[int],
    metrics: tuple[str, ...] = ("goals_achieved", "wasted_steps"),
    n_resamples: int = 2000,
    ci: float = 0.95,
    *, graph: GoalGraph | None = None, log_dir: str | None = None,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Paired comparison: every policy runs against the *same* seeds (docs/10's
    "paired episode seeds across methods"), so any difference reflects the
    policy, not seed-to-seed variance in which interventions/timings
    happened to come up. Returns {policy_name: {metric: (mean, lo, hi)}}.

    `log_dir` (optional, off by default): if given (with `graph`), every
    episode's structured log (see `run_episode`) is appended to
    `{log_dir}/{policy_name}.jsonl` -- one file per policy, so a run's full
    per-goal/per-seed record is on disk alongside the bootstrap summary
    this function returns, not just the aggregate.
    """
    episodes: dict[str, list[dict]] = {name: [] for name in policies}
    for seed in seeds:
        for name, policy_fn in policies.items():
            log_path = f"{log_dir}/{name}.jsonl" if log_dir is not None else None
            episodes[name].append(
                run_episode(env_factory, policy_fn, seed, graph=graph, log_path=log_path)
            )

    return {
        name: {
            metric: bootstrap_ci(
                [float(ep[metric]) for ep in name_episodes], n_resamples=n_resamples, ci=ci,
            )
            for metric in metrics
        }
        for name, name_episodes in episodes.items()
    }
