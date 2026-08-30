#!/usr/bin/env python3
"""Create a claim-auditable final table across visual and state result roots."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_recovery import paired_effect


def state_method(payload, method):
    return next(item for item in payload["environments"] if item["method"] == method)


def visual_method(payload, condition, method=None):
    methods = payload["conditions"][condition]["methods"]
    if method is not None:
        matches = [item for item in methods if item["method"] == method]
        if len(matches) != 1:
            raise ValueError(
                f"candidate aggregate does not contain exactly one {method!r} method"
            )
        return matches[0]
    if len(methods) != 1:
        raise ValueError(
            "multi-method candidate aggregate requires an explicit method selector"
        )
    return methods[0]


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def classify_missing_candidates(candidates):
    missing = list(dict.fromkeys(
        str(item["path"]) for item in candidates if not Path(item["path"]).exists()
    ))
    required = list(dict.fromkeys(
        str(item["path"]) for item in candidates
        if not Path(item["path"]).exists() and item.get("required", True)
    ))
    return missing, required


def _percent(value):
    return f"{100 * float(value):.2f}%"


def _percent_interval(value, interval):
    return f"{_percent(value)} [{_percent(interval[0])}, {_percent(interval[1])}]"


def comparison_markdown(payload):
    lines = [
        "# Held-out V3 visual-control and sweeper-condition comparison", "",
        "| Method | Nominal success | Sweeper-condition raw success | Sweeper-condition safe success | Violations | Protocol interactions / seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    reference = payload["reference"]
    lines.append(
        "| **State PPO reference** | — | "
        f"{_percent(reference['raw_success_rate'])} | "
        f"{_percent(reference['safe_success_rate'])} | "
        f"{_percent(reference['constraint_violation_rate'])} | — |"
    )
    for row in payload["candidates"]:
        consumed = row["protocol_environment_transitions_consumed"]
        if isinstance(consumed, list):
            consumed_text = ", ".join(f"{int(value):,}" for value in consumed)
        else:
            consumed_text = f"{int(consumed):,}"
        lines.append(
            f"| {row['name']} | "
            f"{_percent_interval(row['nominal_success_rate'], row['nominal_success_hierarchical_bootstrap_95'])} | "
            f"{_percent_interval(row['raw_success_rate'], row['raw_success_hierarchical_bootstrap_95'])} | "
            f"{_percent_interval(row['safe_success_rate'], row['safe_success_hierarchical_bootstrap_95'])} | "
            f"{_percent(row['constraint_violation_rate'])} | {consumed_text} |"
        )
    lines.extend([
        "",
        "Intervals are hierarchical 95% bootstrap intervals over training seeds and held-out episodes.",
        "Protocol interactions include the full checkpoint-selection run and inherited DAgger/initializer compute.",
    ])
    if payload.get("required_missing"):
        lines.append(
            "Required missing primary aggregates: "
            + ", ".join(payload["required_missing"]) + "."
        )
    optional_missing = [
        path for path in payload.get("missing", [])
        if path not in payload.get("required_missing", [])
    ]
    if optional_missing:
        lines.append(
            "Optional missing protocol extensions: " + ", ".join(optional_missing) + "."
        )
    lines.append("")
    return "\n".join(lines)


def paired_seed_groups(candidate, reference):
    candidate_by_seed = {
        int(item["training_seed"]): item for item in candidate["seed_results"]
    }
    reference_by_seed = {
        int(item["training_seed"]): item for item in reference["seed_results"]
    }
    if set(candidate_by_seed) != set(reference_by_seed):
        raise ValueError("visual and state methods do not share training seeds")
    left, right = [], []
    for seed in sorted(candidate_by_seed):
        candidate_result = candidate_by_seed[seed]
        reference_result = reference_by_seed[seed]
        if candidate_result["seed_base"] != reference_result["seed_base"]:
            raise ValueError("visual and state methods do not share held-out seed bases")
        candidate_episodes = candidate_result["episode_records"]
        reference_episodes = reference_result["episode_records"]
        if len(candidate_episodes) != len(reference_episodes):
            raise ValueError("visual and state methods have unequal paired episodes")
        for visual_episode, state_episode in zip(candidate_episodes, reference_episodes):
            for branch in ("first_goal_removed", "instruction_red_first"):
                if visual_episode.get(branch) != state_episode.get(branch):
                    raise ValueError(f"held-out branch mismatch for seed {seed}: {branch}")
        left.append(candidate_episodes)
        right.append(reference_episodes)
    return left, right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_comparison_v1.json")
    parser.add_argument("--output", default="results/final_visual_comparison")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected_semantics = config.get(
        "benchmark_semantics", "intervention_target_only_v2"
    )
    reference_payload = json.loads(Path(config["reference"]["path"]).read_text(encoding="utf-8"))
    if reference_payload.get("benchmark_semantics") != expected_semantics:
        raise ValueError("state reference does not match configured benchmark semantics")
    reference = state_method(reference_payload, config["reference"]["method"])
    thresholds = {
        "raw_success_rate": reference["pooled_success_rate"],
        "safe_success_rate": reference["pooled_safe_success_rate"],
        "constraint_violation_rate": reference["constraint_violation_rate"],
    }
    context_baselines = []
    for baseline in config.get("context_baselines", []):
        path = Path(baseline["path"])
        if not path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(path)
        source = json.loads(path.read_text(encoding="utf-8"))
        if baseline["format"] != "nedreamer_pilot":
            raise ValueError(f"unsupported context baseline format: {baseline['format']}")
        aggregate = source["aggregate"]
        context_baselines.append({
            "name": baseline["name"], "algorithm": source["algorithm"],
            "observation_protocol": source["observation_protocol"],
            "training_seeds": aggregate["seed_count"],
            "environment_steps_per_seed": aggregate["environment_steps_per_seed"],
            "evaluation_episodes": aggregate["evaluation_episodes"],
            "success_rate": aggregate["evaluation_success_rate"],
            "zero_success_exact_two_sided_upper_95": aggregate.get(
                "zero_success_exact_two_sided_upper_95"
            ),
            "representation_loss_reduction_fraction_mean": aggregate.get(
                "representation_loss_reduction_fraction_mean"
            ),
            "teleport_control": source["teleport_control"],
            "claim_use": baseline["claim_use"],
        })
    rows = []
    missing, required_missing = classify_missing_candidates(config["candidates"])
    for candidate in config["candidates"]:
        path = Path(candidate["path"])
        if not path.exists():
            continue
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        if aggregate.get("benchmark_semantics") != expected_semantics:
            raise ValueError(f"candidate benchmark semantics mismatch: {path}")
        forced = visual_method(aggregate, "intervention", candidate.get("method"))
        nominal = visual_method(aggregate, "nominal", candidate.get("method"))
        row = {
            "name": candidate["name"], "input_regime": candidate["input_regime"],
            "episodes": forced["episodes"], "seeds": forced["seeds"],
            "raw_success_rate": forced["success_rate"],
            "raw_success_wilson_95": forced["success_wilson_95"],
            "raw_success_hierarchical_bootstrap_95": forced[
                "success_hierarchical_bootstrap_95"
            ],
            "safe_success_rate": forced["safe_success_rate"],
            "safe_success_wilson_95": forced["safe_success_wilson_95"],
            "safe_success_hierarchical_bootstrap_95": forced[
                "safe_success_hierarchical_bootstrap_95"
            ],
            "constraint_violation_rate": forced["constraint_violation_rate"],
            "nominal_success_rate": nominal["success_rate"],
            "nominal_success_wilson_95": nominal["success_wilson_95"],
            "nominal_success_hierarchical_bootstrap_95": nominal[
                "success_hierarchical_bootstrap_95"
            ],
            "nominal_safe_success_rate": nominal["safe_success_rate"],
            "nominal_constraint_violation_rate": nominal[
                "constraint_violation_rate"
            ],
            "online_ppo_environment_steps": forced[
                "online_ppo_environment_steps"
            ],
            "initialization_ppo_environment_steps": forced[
                "initialization_ppo_environment_steps"
            ],
            "ppo_environment_steps": forced["ppo_environment_steps"],
            "online_protocol_ppo_environment_steps": forced[
                "online_protocol_ppo_environment_steps"
            ],
            "initialization_protocol_ppo_environment_steps": forced[
                "initialization_protocol_ppo_environment_steps"
            ],
            "protocol_ppo_environment_steps": forced[
                "protocol_ppo_environment_steps"
            ],
            "local_bc_dagger_environment_transitions": forced[
                "local_bc_dagger_environment_transitions"
            ],
            "initialization_bc_dagger_environment_transitions": forced[
                "initialization_bc_dagger_environment_transitions"
            ],
            "bc_dagger_environment_transitions": forced[
                "bc_dagger_environment_transitions"
            ],
            "total_environment_transitions": forced["total_environment_transitions"],
            "protocol_environment_transitions_consumed": forced[
                "protocol_environment_transitions_consumed"
            ],
            "visual_progress_bit_accuracy": forced.get("visual_progress_bit_accuracy"),
            "visual_progress_exact_accuracy": forced.get("visual_progress_exact_accuracy"),
        }
        candidate_groups, reference_groups = paired_seed_groups(forced, reference)
        row["paired_against_state"] = paired_effect(
            candidate_groups, reference_groups, np.random.default_rng(20260828),
        )
        row["matches_state_raw"] = row["raw_success_rate"] >= thresholds["raw_success_rate"]
        row["matches_state_safe"] = row["safe_success_rate"] >= thresholds["safe_success_rate"]
        row["matches_state_violation"] = row["constraint_violation_rate"] <= thresholds["constraint_violation_rate"]
        row["competitive_on_all_predeclared_metrics"] = all(
            row[key] for key in ("matches_state_raw", "matches_state_safe", "matches_state_violation")
        )
        rows.append(row)
    if required_missing and not args.allow_missing:
        raise FileNotFoundError(
            "missing required final aggregates: " + ", ".join(required_missing)
        )
    payload = {
        "schema_version": 1,
        "protocol": "cross-method comparison on forced-sweeper-condition held-out evaluation",
        "benchmark_semantics": expected_semantics,
        "reference": {"name": config["reference"]["name"], **thresholds},
        "context_baselines": context_baselines,
        "candidates": rows, "missing": missing,
        "required_missing": required_missing,
        "any_candidate_competitive": any(row["competitive_on_all_predeclared_metrics"] for row in rows),
    }
    output = Path(args.output)
    atomic_json(payload, output / "comparison.json")
    columns = [
        "name", "input_regime", "episodes", "seeds", "raw_success_rate",
        "safe_success_rate", "constraint_violation_rate", "nominal_success_rate",
        "online_ppo_environment_steps", "initialization_ppo_environment_steps",
        "ppo_environment_steps", "online_protocol_ppo_environment_steps",
        "initialization_protocol_ppo_environment_steps",
        "protocol_ppo_environment_steps", "local_bc_dagger_environment_transitions",
        "initialization_bc_dagger_environment_transitions",
        "bc_dagger_environment_transitions",
        "total_environment_transitions", "protocol_environment_transitions_consumed",
        "visual_progress_bit_accuracy", "visual_progress_exact_accuracy",
        "matches_state_raw", "matches_state_safe", "matches_state_violation",
        "competitive_on_all_predeclared_metrics",
    ]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)
    markdown_path = output / "comparison.md"
    temporary_markdown = markdown_path.with_name(
        f".{markdown_path.name}.tmp.{os.getpid()}"
    )
    temporary_markdown.write_text(comparison_markdown(payload), encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
