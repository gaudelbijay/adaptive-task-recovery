#!/usr/bin/env python3
"""Combine frozen screening and untouched confirmatory seeds into final results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from aggregate_visual_recovery import (
    hierarchical_binary_interval, paired_effect, succeeded, wilson,
)
from compare_visual_recovery_candidates import state_method, visual_method


def atomic_text(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load(path, semantics):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("benchmark_semantics") != semantics:
        raise ValueError(f"five-seed input has wrong semantics: {path}")
    return payload


def checked_groups(result, expected_seeds, expected_episodes, condition):
    records = sorted(result.get("seed_results", []), key=lambda item: item["training_seed"])
    if [record["training_seed"] for record in records] != sorted(expected_seeds):
        raise ValueError("five-seed input has missing, duplicate, or unexpected seeds")
    if int(result.get("seeds", -1)) != len(expected_seeds):
        raise ValueError("five-seed input reports the wrong seed count")
    if int(result.get("episodes", -1)) != expected_episodes:
        raise ValueError("five-seed input reports the wrong episode count")
    if any(len(record.get("episode_records", [])) != 256 for record in records):
        raise ValueError("five-seed input lacks 256 episode records per policy")
    if any(record.get("condition") != condition for record in records):
        raise ValueError("five-seed input contains a misrouted evaluation condition")
    if any(int(record.get("seed_base", -1)) < 0 for record in records):
        raise ValueError("five-seed input lacks held-out seed-base provenance")
    return records


def common_provenance(records, key):
    values = {
        json.dumps(record.get(key), sort_keys=True) for record in records
    }
    if len(values) != 1 or values == {"null"}:
        raise ValueError(f"five-seed visual records disagree on {key}")
    return records[0][key]


def summary(records, rng):
    groups = [record["episode_records"] for record in records]
    episodes = [episode for group in groups for episode in group]
    success_count = sum(succeeded(episode) for episode in episodes)
    safe_count = sum(
        succeeded(episode) and episode.get("constraint_violated", 0.0) < 0.5
        for episode in episodes
    )
    return {
        "training_seeds": [record["training_seed"] for record in records],
        "seeds": len(records), "episodes": len(episodes),
        "successes": success_count, "success_rate": success_count / len(episodes),
        "success_wilson_95": wilson(success_count, len(episodes)),
        "success_hierarchical_bootstrap_95": hierarchical_binary_interval(
            groups, succeeded, rng,
        ),
        "safe_successes": safe_count,
        "safe_success_rate": safe_count / len(episodes),
        "safe_success_wilson_95": wilson(safe_count, len(episodes)),
        "safe_success_hierarchical_bootstrap_95": hierarchical_binary_interval(
            groups,
            lambda episode: succeeded(episode)
            and episode.get("constraint_violated", 0.0) < 0.5,
            rng,
        ),
        "constraint_violation_rate": float(np.mean([
            episode.get("constraint_violated", 0.0) for episode in episodes
        ])),
        "mean_goals_completed": float(np.mean([
            episode.get("goals_completed", 0.0) for episode in episodes
        ])),
        "seed_success_rates": [
            float(np.mean([succeeded(episode) for episode in group]))
            for group in groups
        ],
        "seed_results": records,
    }


def filtered_groups(records, key, value):
    groups = []
    for record in records:
        selected = [
            episode for episode in record["episode_records"]
            if episode.get(key) == value
        ]
        if not selected:
            raise ValueError(f"five-seed branch is empty: {key}={value}")
        groups.append(selected)
    return groups


def combine(config):
    semantics = config["benchmark_semantics"]
    screen_seeds = list(config["screening_seeds"])
    confirm_seeds = list(config["confirmatory_seeds"])
    if len(screen_seeds) != 3 or len(confirm_seeds) != 2:
        raise ValueError("five-seed protocol requires three screen plus two confirm seeds")
    if set(screen_seeds) & set(confirm_seeds):
        raise ValueError("five-seed screen and confirmation overlap")
    all_seeds = sorted(screen_seeds + confirm_seeds)

    visual_screen_payload = load(config["visual_screening"], semantics)
    visual_confirm_payload = load(config["visual_confirmatory"], semantics)
    state_screen_payload = load(config["state_screening"], semantics)
    state_confirm_payload = load(config["state_confirmatory"], semantics)

    visual_conditions = {}
    visual_records_by_condition = {}
    rng = np.random.default_rng(20260828)
    for condition in ("nominal", "intervention"):
        screen = visual_method(
            visual_screen_payload, condition, config["visual_method"],
        )
        confirm = visual_method(
            visual_confirm_payload, condition, config["visual_method"],
        )
        records = checked_groups(
            screen, screen_seeds, 768, condition,
        ) + checked_groups(
            confirm, confirm_seeds, 512, condition,
        )
        records = sorted(records, key=lambda item: item["training_seed"])
        if [record["training_seed"] for record in records] != all_seeds:
            raise ValueError("visual five-seed merge is incomplete")
        visual_records_by_condition[condition] = records
        visual_conditions[condition] = summary(records, rng)

    state_screen = state_method(state_screen_payload, config["state_method"])
    state_confirm = state_method(state_confirm_payload, config["state_method"])
    state_records = checked_groups(
        state_screen, screen_seeds, 768, "intervention",
    ) + checked_groups(
        state_confirm, confirm_seeds, 512, "intervention",
    )
    state_records = sorted(state_records, key=lambda item: item["training_seed"])
    if [record["training_seed"] for record in state_records] != all_seeds:
        raise ValueError("state five-seed merge is incomplete")
    state_forced = summary(state_records, rng)

    visual_forced_records = visual_records_by_condition["intervention"]
    all_visual_records = [
        record for condition in ("nominal", "intervention")
        for record in visual_records_by_condition[condition]
    ]
    if len({record["seed_base"] for record in all_visual_records + state_records}) != 1:
        raise ValueError("five-seed visual/state records use different held-out seed bases")
    visual_training_source = common_provenance(
        all_visual_records, "training_source_sha256",
    )
    visual_evaluation_source = common_provenance(
        all_visual_records, "evaluation_source_sha256",
    )
    paired = paired_effect(
        [record["episode_records"] for record in visual_forced_records],
        [record["episode_records"] for record in state_records], rng,
    )
    branches = {}
    for value, label in ((1.0, "first_goal_removed"), (0.0, "second_goal_removed")):
        branches[label] = paired_effect(
            filtered_groups(visual_forced_records, "first_goal_removed", value),
            filtered_groups(state_records, "first_goal_removed", value), rng,
        )
    return {
        "schema_version": 1,
        "protocol": "five-seed confirmatory held-out visual/state comparison",
        "benchmark_semantics": semantics,
        "screening_seeds": screen_seeds,
        "confirmatory_seeds": confirm_seeds,
        "all_training_seeds": all_seeds,
        "no_seed_discarded": True,
        "heldout_episodes_per_condition": 1280,
        "heldout_seed_base": all_visual_records[0]["seed_base"],
        "visual_training_source_sha256": visual_training_source,
        "visual_evaluation_source_sha256": visual_evaluation_source,
        "visual_method": config["visual_method"],
        "state_method": config["state_method"],
        "visual": visual_conditions,
        "state_forced_intervention": state_forced,
        "paired_visual_minus_state": paired,
        "paired_branches": branches,
        "claim_boundary": (
            "new seeds were selected before screening recovery completed and are "
            "included regardless of outcome; intervals resample trained policies"
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def markdown(payload):
    forced = payload["visual"]["intervention"]
    nominal = payload["visual"]["nominal"]
    state = payload["state_forced_intervention"]
    paired = payload["paired_visual_minus_state"]
    lines = [
        "# Five-seed confirmatory recovery result", "",
        f"Seeds: {payload['all_training_seeds']} (no seed discarded).", "",
        "| Policy / condition | Raw success | Safe success | Violations |", "|---|---:|---:|---:|",
        f"| Visual / forced | {forced['success_rate']:.2%} | {forced['safe_success_rate']:.2%} | {forced['constraint_violation_rate']:.2%} |",
        f"| Visual / nominal | {nominal['success_rate']:.2%} | {nominal['safe_success_rate']:.2%} | {nominal['constraint_violation_rate']:.2%} |",
        f"| State / forced | {state['success_rate']:.2%} | {state['safe_success_rate']:.2%} | {state['constraint_violation_rate']:.2%} |",
        "", "Paired visual-minus-state forced-intervention effect: "
        f"{paired['safe_success_rate_difference']:+.2%}, hierarchical 95% interval "
        f"[{paired['safe_paired_bootstrap_95'][0]:+.2%}, {paired['safe_paired_bootstrap_95'][1]:+.2%}].",
        "", f"Claim boundary: {payload['claim_boundary']}.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/visual_recovery_five_seed_confirmation_v1.json",
    )
    parser.add_argument("--output", default="results/final_visual_comparison")
    args = parser.parse_args()
    payload = combine(json.loads(Path(args.config).read_text(encoding="utf-8")))
    root = Path(args.output)
    atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", root / "five_seed_confirmation.json")
    atomic_text(markdown(payload), root / "five_seed_confirmation.md")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
