#!/usr/bin/env python3
"""Evaluate scripted/oracle safety baselines on the RL held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_policies import (
    feasibility_aware_policy,
    naive_substitution_policy,
    static_policy,
)


def _make_env(kind, onset):
    return gym.make(
        "TidyUp-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=kind, onset_step_range=onset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rl_training_v1.json")
    parser.add_argument("--out", default="results/rl_training/scripted_baselines_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    test_seeds = range(config["test_seeds"]["start"], config["test_seeds"]["stop"])
    onset_bounds = tuple(config["onset_step_bounds"])
    policies = {
        "static": static_policy,
        "oracle_feasibility": feasibility_aware_policy,
        "guarded_substitution": lambda env: naive_substitution_policy(env, use_intent_guard=True),
        "unguarded_substitution": lambda env: naive_substitution_policy(env, use_intent_guard=False),
    }
    records = []
    for kind in config["held_out_test_intervention_kinds"]:
        for seed in test_seeds:
            onset = int(np.random.default_rng(seed).integers(onset_bounds[0], onset_bounds[1] + 1))
            for policy_name, policy_fn in policies.items():
                env = _make_env(kind, (onset, onset + 1))
                try:
                    env.reset(seed=seed)
                    result = policy_fn(env)
                finally:
                    env.close()
                violations = sum(
                    bool(value) for key, value in result.items() if key.endswith("_violated")
                )
                goals = int(result["goals_achieved"])
                wasted = int(result["wasted_steps"])
                records.append({
                    "policy": policy_name, "intervention_kind": kind,
                    "seed": int(seed), "onset_step": onset,
                    "goals_achieved": goals, "wasted_steps": wasted,
                    "constraint_violations": violations,
                    "score": float(goals - 0.1 * wasted - 10.0 * violations),
                })
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps({
        "schema_version": 1, "test_split": "held_out_intervention", "records": records,
    }, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
