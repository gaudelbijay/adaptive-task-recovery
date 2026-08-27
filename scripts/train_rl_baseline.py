#!/usr/bin/env python3
"""Train one matched non-primary baseline and evaluate its held-out split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal
from atr.language.goal_graph import canonical_example
from atr.policies.domain_randomized import domain_randomized_policy, train_domain_randomized_policy
from atr.policies.imitation import collect_demonstrations, imitation_policy, train_bc_table


def _seed_range(spec: dict) -> list[int]:
    return list(range(int(spec["start"]), int(spec["stop"])))


def _make_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def _evaluate(policy_fn, intervention_kinds, seeds, onset_bounds):
    rows = []
    for kind in intervention_kinds:
        for seed in seeds:
            onset = int(np.random.default_rng(seed).integers(onset_bounds[0], onset_bounds[1] + 1))
            env = _make_env(kind, (onset, onset + 1))
            try:
                env.reset(seed=seed)
                result = policy_fn(env)
            finally:
                env.close()
            goals = int(result["goals_achieved"])
            wasted = int(result["wasted_steps"])
            rows.append({
                "intervention_kind": kind, "seed": int(seed), "onset_step": onset,
                "score": float(goals - 0.1 * wasted),
                "goals_achieved": goals, "wasted_steps": wasted,
            })
    return {
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "mean_goals_achieved": float(np.mean([row["goals_achieved"] for row in rows])),
        "mean_wasted_steps": float(np.mean([row["wasted_steps"] for row in rows])),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_baselines_v1.json")
    parser.add_argument("--output", default="results/rl_training")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = _seed_range(config["seeds"])
    tasks = [(variant, seed) for variant in config["variants"] for seed in seeds]
    if not 0 <= args.task_index < len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")
    variant, seed = tasks[args.task_index]
    if args.preflight:
        print(json.dumps({"tasks": len(tasks), "variant": variant, "seed": seed}, indent=2))
        return

    graph = canonical_example()
    train_kinds = tuple(config["train_intervention_kinds"])
    onset_bounds = tuple(config["onset_step_bounds"])
    episodes = int(config["training_episodes"])
    if variant == "blind_domain_randomized_q":
        table = train_domain_randomized_policy(
            _make_env, graph, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=train_kinds, onset_step_bounds=onset_bounds,
            n_episodes=episodes, seed=seed,
        )
        policy_fn = lambda env: domain_randomized_policy(
            env, table, graph, attempt_goal, _TRAY_SLOTS,
        )
        learned_state = {goal_id: {str(a): float(q) for a, q in values.items()} for goal_id, values in table.items()}
    elif variant == "behavioral_cloning":
        demonstrations = collect_demonstrations(
            _make_env, graph, _TRAY_SLOTS, attempt_goal,
            intervention_kinds=train_kinds, onset_step_bounds=onset_bounds,
            n_episodes=episodes, seed=seed,
        )
        table = train_bc_table(demonstrations)
        policy_fn = lambda env: imitation_policy(env, table, graph, attempt_goal, _TRAY_SLOTS)
        learned_state = {repr(key): int(action) for key, action in table.items()}
    else:
        raise ValueError(f"unsupported baseline {variant!r}")

    test = _evaluate(
        policy_fn, config["held_out_test_intervention_kinds"],
        _seed_range(config["test_seeds"]), onset_bounds,
    )
    run_dir = Path(args.output) / config["name"] / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1, "experiment": config["name"], "variant": variant,
        "seed": seed, "training_episodes": episodes,
        "test_split": "held_out_intervention", "learned_state": learned_state, "test": test,
    }
    temporary = run_dir / ".result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / "result.json")
    print(json.dumps({key: result[key] for key in result if key not in ("test", "learned_state")}, indent=2))


if __name__ == "__main__":
    main()
