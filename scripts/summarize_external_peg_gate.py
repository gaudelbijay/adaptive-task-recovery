#!/usr/bin/env python3
"""Aggregate matched PegInsertion closed-loop manifests against the frozen gate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(successes: int, trials: int, z: float = 1.959963984540054):
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return center - radius, center + radius


def parse_method(value: str):
    name, separator, directory = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("method must be NAME=DIRECTORY")
    return name, Path(directory)


def load(directory: Path):
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    if not records:
        raise RuntimeError(f"no manifests in {directory}")
    return records


def aggregate(records):
    by_condition = defaultdict(
        lambda: {"episodes": 0, "successes": 0, "safe_abstentions": 0, "safe_successes": 0, "violations": 0}
    )
    for record in records:
        row = by_condition[record["condition"]]
        for key in row:
            row[key] += int(record[key])
    overall = {key: sum(row[key] for row in by_condition.values()) for key in next(iter(by_condition.values()))}
    for row in [*by_condition.values(), overall]:
        row["safe_success_rate"] = row["safe_successes"] / row["episodes"]
        row["violation_rate"] = row["violations"] / row["episodes"]
        row["safe_success_wilson_95"] = wilson(row["safe_successes"], row["episodes"])
    by_router_seed = {}
    for seed in sorted({record.get("router_seed") for record in records if record.get("router_seed") is not None}):
        selected = [record for record in records if record.get("router_seed") == seed]
        episodes = sum(int(record["episodes"]) for record in selected)
        safe = sum(int(record["safe_successes"]) for record in selected)
        by_router_seed[str(seed)] = {
            "episodes": episodes,
            "safe_successes": safe,
            "safe_success_rate": safe / episodes,
        }
    return {
        "overall": overall,
        "conditions": dict(sorted(by_condition.items())),
        "by_router_seed": by_router_seed,
    }


def seed_bootstrap_gain(candidate, baseline, samples=10000, seed=20260831):
    common = sorted(set(candidate["by_router_seed"]) & set(baseline["by_router_seed"]))
    if len(common) < 2:
        return None
    gains = [
        candidate["by_router_seed"][key]["safe_success_rate"]
        - baseline["by_router_seed"][key]["safe_success_rate"]
        for key in common
    ]
    rng = random.Random(seed)
    draws = sorted(
        sum(gains[rng.randrange(len(gains))] for _ in gains) / len(gains)
        for _ in range(samples)
    )
    return {
        "paired_router_seeds": [int(key) for key in common],
        "gain_by_seed": gains,
        "bootstrap_samples": samples,
        "bootstrap_95": [draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1]],
    }


def paired_episode_cluster_bootstrap_gain(
    candidate_records, baseline_records, samples=10000, seed=20260901,
):
    """Bootstrap router seeds and matched physical episodes as two clusters."""
    def matrices(records):
        by_router = defaultdict(dict)
        for record in records:
            router = record.get("router_seed")
            key = "shared" if router is None else str(router)
            episode_ids = record.get("episode_ids", [])
            outcomes = record.get("episode_safe_outcome", [])
            if len(episode_ids) != len(outcomes):
                raise RuntimeError("episode IDs/outcomes are missing or misaligned")
            for episode_id, outcome in zip(episode_ids, outcomes):
                if episode_id in by_router[key]:
                    raise RuntimeError(f"duplicate episode {episode_id} for router {key}")
                by_router[key][episode_id] = float(outcome)
        return dict(by_router)

    candidate = matrices(candidate_records)
    baseline = matrices(baseline_records)
    candidate_keys = sorted(key for key in candidate if key != "shared")
    if len(candidate_keys) < 2:
        return None
    if "shared" in baseline:
        baseline_keys = ["shared"] * len(candidate_keys)
    elif set(candidate_keys).issubset(baseline):
        baseline_keys = candidate_keys
    else:
        return None
    common_episodes = sorted(set.intersection(*(
        *[set(candidate[key]) for key in candidate_keys],
        *[set(baseline[key]) for key in set(baseline_keys)],
    )))
    if not common_episodes:
        raise RuntimeError("methods have no common episode IDs")
    candidate_matrix = np.asarray([
        [candidate[key][episode] for episode in common_episodes]
        for key in candidate_keys
    ], dtype=np.float64)
    baseline_matrix = np.asarray([
        [baseline[key][episode] for episode in common_episodes]
        for key in baseline_keys
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    router_count, episode_count = candidate_matrix.shape
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        router_index = rng.integers(0, router_count, size=(count, router_count))
        episode_index = rng.integers(0, episode_count, size=(count, episode_count))
        candidate_draw = candidate_matrix[
            router_index[:, :, None], episode_index[:, None, :]
        ].mean(axis=(1, 2))
        baseline_draw = baseline_matrix[
            router_index[:, :, None], episode_index[:, None, :]
        ].mean(axis=(1, 2))
        draws.extend((candidate_draw - baseline_draw).tolist())
    draws.sort()
    return {
        "paired_router_seeds": [int(key) for key in candidate_keys],
        "common_episode_ids": len(common_episodes),
        "bootstrap_samples": samples,
        "point_gain": float((candidate_matrix - baseline_matrix).mean()),
        "bootstrap_95": [
            draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--method", action="append", type=parse_method, required=True)
    parser.add_argument("--candidate", default="causal")
    parser.add_argument("--oracle", action="append", default=["oracle"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text())
    criteria = gate["pass_criteria"]
    grouped = defaultdict(list)
    for name, directory in args.method:
        grouped[name].extend(load(directory))
    methods = {name: aggregate(records) for name, records in grouped.items()}
    candidate = methods[args.candidate]
    overall = candidate["overall"]
    conditions = candidate["conditions"]
    checks = {
        "closed_loop_safe_recovery": overall["safe_success_rate"] >= criteria["closed_loop_safe_recovery_min"],
        "native_nominal_success": conditions["nominal"]["safe_success_rate"] >= criteria["native_nominal_success_min"],
        "heldout_ejection_safe_recovery": conditions["negative_lateral_peg_ejection"]["safe_success_rate"] >= criteria["heldout_ejection_safe_recovery_min"],
        "permanent_safe_abstention": conditions["permanent_hole_block"]["safe_success_rate"] >= criteria["permanent_safe_abstention_min"],
        "violation_rate": overall["violation_rate"] <= criteria["violation_rate_max"],
    }
    router_seeds = {
        record["router_seed"] for record in grouped[args.candidate]
        if record.get("router_seed") is not None
    }
    checks["training_seed_count"] = len(router_seeds) >= criteria["minimum_independent_training_seeds"]
    competitors = {
        name: method for name, method in methods.items()
        if name != args.candidate and name not in set(args.oracle)
    }
    strongest_name, strongest = max(
        competitors.items(), key=lambda item: item[1]["overall"]["safe_success_rate"],
    )
    baseline = strongest["overall"]
    gain = overall["safe_success_rate"] - baseline["safe_success_rate"]
    difference_ci = (
        overall["safe_success_wilson_95"][0] - baseline["safe_success_wilson_95"][1],
        overall["safe_success_wilson_95"][1] - baseline["safe_success_wilson_95"][0],
    )
    checks["gain_over_strongest_non_oracle"] = gain >= criteria["gain_over_strongest_non_oracle_min_pp"] / 100
    checks["gain_newcombe_lower"] = difference_ci[0] > criteria["gain_newcombe_95_lower_min_pp"] / 100
    hierarchical_gain = seed_bootstrap_gain(candidate, strongest)
    paired_cluster_gain = paired_episode_cluster_bootstrap_gain(
        grouped[args.candidate], grouped[strongest_name],
    )
    if "gain_cluster_bootstrap_95_lower_min_pp" in criteria:
        checks["gain_cluster_bootstrap_lower"] = (
            paired_cluster_gain is not None
            and paired_cluster_gain["bootstrap_95"][0]
            > criteria["gain_cluster_bootstrap_95_lower_min_pp"] / 100
        )
    result = {
        "schema_version": 1,
        "gate": str(args.gate),
        "candidate": args.candidate,
        "methods": methods,
        "comparison": {
            "strongest_non_oracle": strongest_name,
            "gain": gain,
            "newcombe_95": difference_ci,
            "training_seed_bootstrap": hierarchical_gain,
            "paired_episode_cluster_bootstrap": paired_cluster_gain,
        },
        "checks": checks,
        "external_gate_pass": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
