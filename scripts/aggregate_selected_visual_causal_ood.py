#!/usr/bin/env python3
"""Aggregate paired selected-policy causal-head and visual-OOD evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from run_selected_visual_causal_ood import evaluation_filename, load, resolve_task


EVAL_PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def success(record: dict) -> bool:
    for key in ("success_once", "success_at_end", "success"):
        if key in record:
            return float(record[key]) >= 0.5
    raise ValueError("episode record lacks a success metric")


def safe_success(record: dict) -> float:
    return float(success(record) and float(record.get("constraint_violated", 0.0)) < 0.5)


def paired_cluster_interval(
    groups: list[np.ndarray], rng: np.random.Generator, repetitions: int,
) -> tuple[float, list[float]]:
    if not groups or any(group.ndim != 1 or not len(group) for group in groups):
        raise ValueError("paired bootstrap requires non-empty one-dimensional groups")
    if len({len(group) for group in groups}) != 1:
        raise ValueError("paired bootstrap requires equal episode counts per seed")
    values = np.stack(groups)
    seed_count, episode_count = values.shape
    totals = np.zeros(repetitions, dtype=float)
    for _ in range(seed_count):
        sampled_seeds = rng.integers(0, seed_count, size=repetitions)
        sampled_episodes = rng.integers(
            0, episode_count, size=(repetitions, episode_count),
        )
        totals += values[
            sampled_seeds[:, None], sampled_episodes,
        ].sum(axis=1)
    bootstrap = totals / (seed_count * episode_count)
    return float(values.mean()), np.quantile(bootstrap, [0.025, 0.975]).tolist()


def validate_eval(
    payload: dict, *, method: str, seed: int, condition: str,
    episodes: int, checkpoint_sha256: str,
) -> None:
    if payload.get("protocol") != EVAL_PROTOCOL:
        raise ValueError("evaluation has the wrong protocol")
    if payload.get("method") != method or int(payload.get("training_seed", -1)) != seed:
        raise ValueError("evaluation policy identity mismatch")
    if payload.get("condition") != condition or int(payload.get("episodes", -1)) != episodes:
        raise ValueError("evaluation condition or episode count mismatch")
    if len(payload.get("episode_records", [])) != episodes:
        raise ValueError("evaluation episode-record count mismatch")
    observed_hash = payload.get("checkpoint_sha256")
    if observed_hash is not None and observed_hash != checkpoint_sha256:
        raise ValueError("evaluation checkpoint hash mismatch")


def aggregate(spec_path: str | Path, results_root: str | Path) -> dict:
    spec_path = Path(spec_path)
    spec = load(spec_path)
    selection_path = Path(spec["selection"])
    selection = load(selection_path)
    first = resolve_task(spec, selection, 0)
    policy = first["policy"]
    experiment = policy["experiments"][0]
    method = experiment["method"]
    episodes = int(spec["episodes"])
    seeds = [int(seed) for seed in policy["seeds"]]
    thresholds = spec["hypothesis_thresholds"]
    records = []
    source_hashes = {
        str(spec_path): sha256(spec_path), str(selection_path): sha256(selection_path),
        first["policy_config"]: sha256(first["policy_config"]),
    }
    baseline_by_condition: dict[str, list[np.ndarray]] = {
        condition: [] for condition in spec["conditions"]
    }
    baseline_payloads: dict[tuple[int, str], dict] = {}
    root = Path(results_root) / policy["name"] / method
    for seed in seeds:
        run_dir = root / f"seed_{seed}"
        checkpoint_hash = sha256(run_dir / "best.pt")
        for condition in spec["conditions"]:
            baseline_path = run_dir / evaluation_filename(
                condition, "normal", "none", "nominal",
            )
            baseline = load(baseline_path)
            validate_eval(
                baseline, method=method, seed=seed, condition=condition,
                episodes=episodes, checkpoint_sha256=checkpoint_hash,
            )
            source_hashes[str(baseline_path)] = sha256(baseline_path)
            baseline_payloads[(seed, condition)] = baseline
            baseline_by_condition[condition].append(np.asarray([
                safe_success(item) for item in baseline["episode_records"]
            ]))

    rng = np.random.default_rng(int(thresholds["bootstrap_seed"]))
    repetitions = int(thresholds["bootstrap_repetitions"])
    for variant in spec["variants"]:
        for condition_index, condition in enumerate(spec["conditions"]):
            paired_groups = []
            variant_groups = []
            per_seed = []
            for seed_index, seed in enumerate(seeds):
                run_dir = root / f"seed_{seed}"
                checkpoint_hash = sha256(run_dir / "best.pt")
                path = run_dir / evaluation_filename(
                    condition, variant["progress_head_mode"],
                    variant["visual_perturbation"],
                    variant.get("environment_profile", "nominal"),
                )
                payload = load(path)
                validate_eval(
                    payload, method=method, seed=seed, condition=condition,
                    episodes=episodes, checkpoint_sha256=checkpoint_hash,
                )
                if payload.get("progress_head_mode") != variant["progress_head_mode"]:
                    raise ValueError("progress-head variant mismatch")
                if payload.get("visual_perturbation") != variant["visual_perturbation"]:
                    raise ValueError("sensor-space variant mismatch")
                if payload.get("environment_profile") != variant.get("environment_profile", "nominal"):
                    raise ValueError("rendered-environment variant mismatch")
                baseline = baseline_payloads[(seed, condition)]
                if payload.get("batch_seeds") != baseline.get("batch_seeds"):
                    raise ValueError("variant and baseline episode seeds are not paired")
                values = np.asarray([
                    safe_success(item) for item in payload["episode_records"]
                ])
                baseline_values = baseline_by_condition[condition][seed_index]
                difference = baseline_values - values
                paired_groups.append(difference)
                variant_groups.append(values)
                per_seed.append({
                    "training_seed": seed,
                    "baseline_safe_success_rate": float(baseline_values.mean()),
                    "variant_safe_success_rate": float(values.mean()),
                    "baseline_minus_variant_safe_success": float(difference.mean()),
                })
                source_hashes[str(path)] = sha256(path)
            mean_drop, interval = paired_cluster_interval(
                paired_groups, rng, repetitions,
            )
            variant_safe = float(np.concatenate(variant_groups).mean())
            record = {
                "variant": variant["name"], "condition": condition,
                "progress_head_mode": variant["progress_head_mode"],
                "visual_perturbation": variant["visual_perturbation"],
                "environment_profile": variant.get("environment_profile", "nominal"),
                "episodes": episodes * len(seeds), "training_seeds": seeds,
                "variant_safe_success_rate": variant_safe,
                "baseline_safe_success_rate": float(np.concatenate(
                    baseline_by_condition[condition]
                ).mean()),
                "baseline_minus_variant_safe_success": mean_drop,
                "paired_cluster_bootstrap_95": interval,
                "per_seed": per_seed,
            }
            records.append(record)

    primary = thresholds["primary_causal_variant"]
    primary_condition = thresholds["primary_causal_condition"]
    primary_records = [
        item for item in records
        if item["variant"] == primary and item["condition"] == primary_condition
    ]
    causal_confirmed = bool(primary_records) and all(
        item["baseline_minus_variant_safe_success"]
        >= float(thresholds["minimum_causal_safe_success_drop"])
        and item["paired_cluster_bootstrap_95"][0] > 0
        for item in primary_records
    )
    ood_records = [
        item for item in records if not item["variant"].startswith("progress_")
    ]
    ood_robust = bool(ood_records) and all(
        item["variant_safe_success_rate"]
        >= float(thresholds["minimum_ood_safe_success"])
        and item["paired_cluster_bootstrap_95"][1]
        <= float(thresholds["maximum_ood_safe_success_drop"])
        for item in ood_records
    )
    if any(not math.isfinite(value) for item in records for value in (
        item["variant_safe_success_rate"],
        item["baseline_minus_variant_safe_success"],
        *item["paired_cluster_bootstrap_95"],
    )):
        raise ValueError("aggregate contains non-finite statistics")
    return {
        "schema_version": 1,
        "protocol": "paired selected-policy causal-head and visual-OOD aggregate",
        "selected": first["selected"], "method": method,
        "training_seeds": seeds, "conditions": spec["conditions"],
        "thresholds": thresholds, "records": records,
        "hypotheses": {
            "learned_progress_head_has_causal_control_utility": causal_confirmed,
            "selected_policy_is_robust_to_frozen_visual_ood_suite": ood_robust,
        },
        "source_sha256": source_hashes,
        "claim_boundary": spec["claim_boundary"],
    }


def markdown(payload: dict) -> str:
    lines = [
        "# Selected visual policy: causal head and OOD evaluation", "",
        f"Selected policy: `{payload['selected']}`.", "",
        "| Variant | Condition | Safe success | Baseline | Baseline - variant (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for item in payload["records"]:
        low, high = item["paired_cluster_bootstrap_95"]
        lines.append(
            f"| {item['variant']} | {item['condition']} | "
            f"{100 * item['variant_safe_success_rate']:.2f}% | "
            f"{100 * item['baseline_safe_success_rate']:.2f}% | "
            f"{100 * item['baseline_minus_variant_safe_success']:+.2f} "
            f"[{100 * low:+.2f}, {100 * high:+.2f}] pp |"
        )
    lines.extend(["", f"Claim boundary: {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", default="results/visual_recovery_ppo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = aggregate(args.config, args.results_root)
    payload["aggregator_sha256"] = sha256(Path(__file__))
    output = Path(args.output)
    atomic_write(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    atomic_write(output.with_suffix(".md"), markdown(payload))
    print(json.dumps({
        "selected": payload["selected"], "hypotheses": payload["hypotheses"],
        "records": len(payload["records"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
