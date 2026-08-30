#!/usr/bin/env python3
"""Fail-closed aggregation for matched physical-removal evaluations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from aggregate_visual_recovery import (
    PROTOCOL as BASE_VISUAL_PROTOCOL,
    hierarchical_binary_interval,
    paired_effect,
    succeeded,
    validate_record as validate_visual_record,
    wilson,
)
from evaluation_seed import validate_record_batch_seeds


RESULT_FILENAME = "heldout_eval_strict_intervention.json"


def _load_training_cohort(spec: dict) -> tuple[dict, dict]:
    config = json.loads(Path(spec["config"]).read_text(encoding="utf-8"))
    if len(config["experiments"]) != 1:
        raise ValueError(f"strict cohort {spec['label']} must declare one method")
    return config, config["experiments"][0]


def _validate_common(
    record: dict, *, strict: dict, strict_hash: str, experiment: dict,
    seed: int, episodes: int,
) -> None:
    expected = {
        "protocol": strict["protocol"],
        "condition": "strict_intervention",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "env_id": experiment["env_id"],
        "method": experiment["method"],
        "training_seed": seed,
        "checkpoint": "best.pt",
        "seed_base": strict["seed_base"],
        "episodes": episodes,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"strict record {key} mismatch: expected {value!r}, "
                f"observed {record.get(key)!r}"
            )
    episode_records = record.get("episode_records")
    if not isinstance(episode_records, list) or len(episode_records) != episodes:
        raise ValueError("strict record has the wrong episode count")
    required = set(strict["required_episode_fields"])
    for index, episode in enumerate(episode_records):
        missing = required - episode.keys()
        if missing:
            raise ValueError(f"strict episode {index} lacks {sorted(missing)}")
        if episode["intervention_occurred"] < 0.5:
            raise ValueError(f"strict episode {index} lacks an intervention")
        if episode["goals_unavailable"] < 0.5:
            raise ValueError(f"strict episode {index} lacks physical removal")
        succeeded(episode)
    validate_record_batch_seeds(record, episodes)
    strict_block = record.get("strict_removal", {})
    if int(strict_block.get("actual_unavailable_episodes", -1)) != episodes:
        raise ValueError("strict physical-removal count is inconsistent")
    if strict_block.get("intervention_overrides") != strict["intervention_overrides"]:
        raise ValueError("strict intervention parameters do not match locked config")
    sources = record.get("evaluation_source_sha256", {})
    if sources.get("strict_config") != strict_hash:
        raise ValueError("strict result does not bind the locked config hash")
    if not sources.get("strict_evaluator") or not sources.get("evaluator"):
        raise ValueError("strict result lacks evaluator provenance")
    successes = sum(succeeded(item) for item in episode_records)
    safe = sum(
        succeeded(item) and item["constraint_violated"] < 0.5
        for item in episode_records
    )
    if int(record.get("successes", -1)) != successes:
        raise ValueError("strict success count is inconsistent")
    if int(record.get("safe_successes", -1)) != safe:
        raise ValueError("strict safe-success count is inconsistent")
    if not math.isclose(float(record.get("success_rate", -1)), successes / episodes):
        raise ValueError("strict success rate is inconsistent")
    if not math.isclose(float(record.get("safe_success_rate", -1)), safe / episodes):
        raise ValueError("strict safe-success rate is inconsistent")


def _validate_record(
    record: dict, *, kind: str, strict: dict, strict_hash: str,
    experiment: dict, seed: int, episodes: int,
) -> None:
    _validate_common(
        record, strict=strict, strict_hash=strict_hash,
        experiment=experiment, seed=seed, episodes=episodes,
    )
    if kind == "visual":
        base_view = copy.deepcopy(record)
        base_view["protocol"] = BASE_VISUAL_PROTOCOL
        base_view["condition"] = "intervention"
        validate_visual_record(
            base_view, experiment, seed, "intervention", episodes,
        )
    elif kind == "state":
        if not record.get("checkpoint_file_sha256"):
            raise ValueError("state strict result lacks immutable checkpoint hash")
        expected_task_hash = hashlib.sha256(
            json.dumps(
                {**experiment, "seed": seed},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if record.get("training_task_sha256") != expected_task_hash:
            raise ValueError("state strict result has the wrong task hash")
        if int(record.get("checkpoint_global_step", -1)) <= 0:
            raise ValueError("state strict result has an invalid checkpoint step")
    else:
        raise ValueError(f"unknown strict cohort kind: {kind}")


def _branch_summary(seed_groups: list[list[dict]], first_removed: bool, rng) -> dict:
    groups = [
        [item for item in group if bool(item["first_goal_removed"] >= 0.5) == first_removed]
        for group in seed_groups
    ]
    episodes = [item for group in groups for item in group]
    raw = sum(succeeded(item) for item in episodes)
    safe = sum(
        succeeded(item) and item["constraint_violated"] < 0.5 for item in episodes
    )
    return {
        "episodes": len(episodes),
        "successes": raw,
        "success_rate": raw / len(episodes),
        "success_wilson_95": wilson(raw, len(episodes)),
        "success_hierarchical_bootstrap_95": hierarchical_binary_interval(
            groups, succeeded, rng,
        ),
        "safe_successes": safe,
        "safe_success_rate": safe / len(episodes),
        "safe_success_wilson_95": wilson(safe, len(episodes)),
        "safe_success_hierarchical_bootstrap_95": hierarchical_binary_interval(
            groups,
            lambda item: succeeded(item) and item["constraint_violated"] < 0.5,
            rng,
        ),
    }


def _filter_branch(seed_groups: list[list[dict]], first_removed: bool) -> list[list[dict]]:
    groups = [
        [item for item in group if bool(item["first_goal_removed"] >= 0.5) == first_removed]
        for group in seed_groups
    ]
    if any(not group for group in groups):
        raise ValueError("strict branch comparison has an empty training-seed group")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/strict_removal_comparison_v1.json",
    )
    parser.add_argument("--output", default="results/strict_removal_comparison")
    args = parser.parse_args()
    comparison = json.loads(Path(args.config).read_text(encoding="utf-8"))
    strict_path = Path(comparison["strict_config"])
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    strict_hash = hashlib.sha256(strict_path.read_bytes()).hexdigest()
    episodes_per_seed = int(strict["episodes_per_training_seed"])
    rng = np.random.default_rng(20260828)
    cohorts = {}
    record_groups = {}
    seed_orders = {}
    batch_seeds_by_seed = {}

    for spec in comparison["cohorts"]:
        config, experiment = _load_training_cohort(spec)
        method = experiment["method"]
        root = Path(spec["output"]) / config["name"] / method
        records = []
        for seed in config["seeds"]:
            path = root / f"seed_{seed}" / RESULT_FILENAME
            record = json.loads(path.read_text(encoding="utf-8"))
            _validate_record(
                record, kind=spec["kind"], strict=strict,
                strict_hash=strict_hash, experiment=experiment,
                seed=int(seed), episodes=episodes_per_seed,
            )
            prior = batch_seeds_by_seed.setdefault(int(seed), record["batch_seeds"])
            if record["batch_seeds"] != prior:
                raise ValueError("matched cohorts do not use identical batch seeds")
            records.append(record)
        if len({
            json.dumps(record["evaluation_source_sha256"], sort_keys=True)
            for record in records
        }) != 1:
            raise ValueError(f"cohort {spec['label']} mixes evaluator provenance")
        groups = [record["episode_records"] for record in records]
        flat = [item for group in groups for item in group]
        raw = sum(succeeded(item) for item in flat)
        safe = sum(
            succeeded(item) and item["constraint_violated"] < 0.5 for item in flat
        )
        cohorts[spec["label"]] = {
            "label": spec["label"], "kind": spec["kind"], "method": method,
            "training_config": config["name"], "training_seeds": config["seeds"],
            "episodes": len(flat), "successes": raw,
            "success_rate": raw / len(flat),
            "success_wilson_95": wilson(raw, len(flat)),
            "success_hierarchical_bootstrap_95": hierarchical_binary_interval(
                groups, succeeded, rng,
            ),
            "safe_successes": safe, "safe_success_rate": safe / len(flat),
            "safe_success_wilson_95": wilson(safe, len(flat)),
            "safe_success_hierarchical_bootstrap_95": hierarchical_binary_interval(
                groups,
                lambda item: succeeded(item) and item["constraint_violated"] < 0.5,
                rng,
            ),
            "constraint_violation_rate": float(np.mean([
                item["constraint_violated"] for item in flat
            ])),
            "constraint_violation_wilson_95": wilson(
                sum(item["constraint_violated"] >= 0.5 for item in flat), len(flat),
            ),
            "constraint_violation_hierarchical_bootstrap_95": (
                hierarchical_binary_interval(
                    groups, lambda item: item["constraint_violated"] >= 0.5, rng,
                )
            ),
            "mean_goals_completed": float(np.mean([
                item.get("goals_completed", 0.0) for item in flat
            ])),
            "first_goal_physically_removed": _branch_summary(groups, True, rng),
            "second_goal_physically_removed": _branch_summary(groups, False, rng),
            "seed_success_rates": [record["success_rate"] for record in records],
            "seed_safe_success_rates": [record["safe_success_rate"] for record in records],
            "checkpoint_global_steps": [
                record["checkpoint_global_step"] for record in records
            ],
            "evaluation_source_sha256": records[0]["evaluation_source_sha256"],
        }
        record_groups[spec["label"]] = groups
        seed_orders[spec["label"]] = [int(seed) for seed in config["seeds"]]

    comparisons = []
    branch_comparisons = {
        "first_goal_physically_removed": [],
        "second_goal_physically_removed": [],
    }
    for left, right in itertools.combinations(cohorts, 2):
        if seed_orders[left] != seed_orders[right]:
            raise ValueError("paired cohorts have different training-seed order")
        result = {"left": left, "right": right}
        result.update(paired_effect(record_groups[left], record_groups[right], rng))
        comparisons.append(result)
        for first_removed, branch in (
            (True, "first_goal_physically_removed"),
            (False, "second_goal_physically_removed"),
        ):
            branch_result = {"left": left, "right": right}
            branch_result.update(paired_effect(
                _filter_branch(record_groups[left], first_removed),
                _filter_branch(record_groups[right], first_removed), rng,
            ))
            branch_comparisons[branch].append(branch_result)
    payload = {
        "schema_version": 1,
        "experiment": comparison["name"],
        "protocol": strict["protocol"],
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "strict_config": str(strict_path),
        "strict_config_sha256": strict_hash,
        "seed_base": strict["seed_base"],
        "episodes_per_training_seed": episodes_per_seed,
        "intervention_overrides": strict["intervention_overrides"],
        "claim_boundary": strict["claim_boundary"],
        "protocol_calibration": strict.get("protocol_calibration"),
        "cohorts": list(cohorts.values()),
        "paired_comparisons": comparisons,
        "paired_comparisons_by_branch": branch_comparisons,
    }
    output = Path(args.output) / comparison["name"] / "aggregate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
