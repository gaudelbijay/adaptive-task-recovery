#!/usr/bin/env python3
"""Train/evaluate one resumable RL seed selected from a Slurm array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401  (registers TidyUp-v1)
from atr.envs.tidy_up_policies import _TRAY_SLOTS, attempt_goal
from atr.language.goal_graph import canonical_example
from atr.policies.q_learning import learned_policy, load_q_table_checkpoint, train_q_table


def _seed_range(spec: dict) -> list[int]:
    return list(range(int(spec["start"]), int(spec["stop"])))


def _make_env(intervention_kind: str, onset_step_range: tuple[int, int]):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )


def _policy_score(result: dict) -> float:
    # Same goal credit and failed-attempt cost used by q_learning.py.
    return float(result["goals_achieved"] - 0.1 * result["wasted_steps"])


def _evaluate(
    q_table: dict,
    intervention_kinds: list[str],
    seeds: list[int],
    onset_step_bounds: tuple[int, int],
    include_intervention_kind: bool,
) -> dict:
    graph = canonical_example()
    rows = []
    for intervention_kind in intervention_kinds:
        for seed in seeds:
            # Let each paired seed choose a deterministic single onset from
            # the declared interval while keeping methods directly matched.
            rng = np.random.default_rng(seed)
            onset = int(rng.integers(onset_step_bounds[0], onset_step_bounds[1] + 1))
            env = _make_env(intervention_kind, (onset, onset + 1))
            try:
                env.reset(seed=seed)
                result = learned_policy(
                    env, q_table, graph, attempt_goal, _TRAY_SLOTS,
                    include_intervention_kind=include_intervention_kind,
                )
            finally:
                env.close()
            rows.append({
                "intervention_kind": intervention_kind,
                "seed": int(seed),
                "onset_step": onset,
                "score": _policy_score(result),
                "goals_achieved": int(result["goals_achieved"]),
                "wasted_steps": int(result["wasted_steps"]),
            })
    return {
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "mean_goals_achieved": float(np.mean([row["goals_achieved"] for row in rows])),
        "mean_wasted_steps": float(np.mean([row["wasted_steps"] for row in rows])),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_training_v1.json")
    parser.add_argument("--output", default="results/rl_training")
    parser.add_argument(
        "--task-index", type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")),
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("only RL training schema_version=1 is supported")
    seeds = _seed_range(config["seeds"])
    tasks = [(variant, seed) for variant in config["variants"] for seed in seeds]
    if not 0 <= args.task_index < len(tasks):
        raise ValueError(f"task-index must be in [0, {len(tasks) - 1}]")
    variant, seed = tasks[args.task_index]
    if args.preflight:
        print(json.dumps({
            "name": config["name"], "tasks": len(tasks), "task_index": args.task_index,
            "variant": variant["name"], "seed": seed,
            "episodes": config["episodes"],
        }, indent=2))
        return

    run_dir = Path(args.output) / config["name"] / variant["name"] / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    graph = canonical_example()
    onset_bounds = tuple(config["onset_step_bounds"])
    include_kind = bool(variant["include_intervention_kind"])
    validation_seeds = _seed_range(config["validation_seeds"])

    def validation_fn(q_table: dict) -> float:
        report = _evaluate(
            q_table, list(config["train_intervention_kinds"]), validation_seeds,
            onset_bounds, include_kind,
        )
        # Retain each checkpoint's validation curve independently of best/latest.
        history_path = run_dir / "validation_history.jsonl"
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "mean_score": report["mean_score"],
                "mean_goals_achieved": report["mean_goals_achieved"],
                "mean_wasted_steps": report["mean_wasted_steps"],
            }) + "\n")
        return report["mean_score"]

    train_q_table(
        make_env=_make_env,
        graph=graph,
        tray_slots=_TRAY_SLOTS,
        attempt_goal_fn=attempt_goal,
        intervention_kinds=tuple(config["train_intervention_kinds"]),
        onset_step_bounds=onset_bounds,
        n_episodes=int(config["episodes"]),
        seed=seed,
        include_intervention_kind=include_kind,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=int(config["checkpoint_every"]),
        resume=True,
        validation_fn=validation_fn,
    )

    best_q = load_q_table_checkpoint(checkpoint_dir / "best.json")
    test_report = _evaluate(
        best_q, list(config["held_out_test_intervention_kinds"]),
        _seed_range(config["test_seeds"]), onset_bounds, include_kind,
    )
    result = {
        "schema_version": 1,
        "experiment": config["name"],
        "variant": variant["name"],
        "seed": seed,
        "training_episodes": int(config["episodes"]),
        "best_checkpoint": str(checkpoint_dir / "best.json"),
        "test_split": "held_out_intervention",
        "test": test_report,
    }
    temporary = run_dir / ".result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, run_dir / "result.json")
    print(json.dumps({key: result[key] for key in result if key != "test"}, indent=2))


if __name__ == "__main__":
    main()
