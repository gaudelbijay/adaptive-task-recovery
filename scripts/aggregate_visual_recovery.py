#!/usr/bin/env python3
"""Aggregate visual-policy evaluations with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

from evaluation_seed import validate_record_batch_seeds


PROTOCOL = "held-out deterministic restricted-input visual-policy evaluation"
BRANCH_KEYS = ("first_goal_removed", "instruction_red_first")


def wilson(successes, trials, z=1.959963984540054):
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [center - radius, center + radius]


def succeeded(record):
    for key in ("success_once", "success_at_end", "success"):
        if key in record:
            return record[key] >= 0.5
    raise KeyError("episode has no success field")


def expected_observation_contract(experiment):
    if experiment.get("actor_learned_goal_progress", False):
        return "rgb_robot_proprio_instruction_visual_progress_v4"
    if experiment.get("actor_goal_progress", False):
        return "rgb_robot_proprio_instruction_progress_v3"
    if experiment.get("actor_tcp_pose", False):
        return "rgb_qpos_qvel_tcp_instruction_v2"
    return "rgb_qpos_qvel_instruction_v1"


def configured_ppo_budget(experiment):
    batch = int(experiment["num_envs"]) * int(experiment["num_steps"])
    # The trainer executes only complete rollout batches:
    # ``iterations = total_timesteps // batch``.
    return (int(experiment["total_timesteps"]) // batch) * batch


def transition_accounting_view(record, experiment):
    """Validate non-PPO accounting, then provide a legacy-equivalent view."""

    if record.get("ppo_accounting_applicable", True) is not False:
        return record
    protocol_fields = {
        "supervised_translation_repair_v34": "translation_training_transitions",
    }
    protocol = record.get("training_protocol")
    if protocol not in protocol_fields:
        raise ValueError(f"unsupported non-PPO training protocol: {protocol!r}")
    zero_fields = (
        "online_ppo_environment_steps",
        "initialization_ppo_environment_steps",
        "ppo_environment_steps",
        "online_protocol_ppo_environment_steps",
        "initialization_protocol_ppo_environment_steps",
        "protocol_ppo_environment_steps",
        "local_bc_dagger_environment_transitions",
        "initialization_bc_dagger_environment_transitions",
        "bc_dagger_environment_transitions",
    )
    if any(int(record.get(field, -1)) != 0 for field in zero_fields):
        raise ValueError("non-PPO evaluation reports PPO or BC/DAgger transitions")
    local = int(record.get(protocol_fields[protocol], -1))
    checkpoint_step = int(record.get("checkpoint_global_step", -2))
    if local != checkpoint_step or local != configured_ppo_budget(experiment):
        raise ValueError("non-PPO local budget disagrees with checkpoint/configuration")
    initialization = int(record.get("initialization_simulator_transitions", -1))
    total = int(record.get("total_environment_transitions", -1))
    consumed = int(record.get("protocol_environment_transitions_consumed", -1))
    if initialization < 0 or total != initialization + local or consumed != total:
        raise ValueError("non-PPO simulator-transition accounting is inconsistent")
    if record.get("initialization_provenance") is not None:
        raise ValueError("non-PPO adapter must not report PPO initialization provenance")

    # The remainder of the long-standing validator checks a PPO-shaped view.
    # This copy is created only after validating the immutable non-PPO fields;
    # it never changes the source evaluation record or its reported accounting.
    view = dict(record)
    view.update({
        "online_ppo_environment_steps": local,
        "online_protocol_ppo_environment_steps": local,
        "ppo_environment_steps": local,
        "protocol_ppo_environment_steps": local,
        "total_environment_transitions": local,
        "protocol_environment_transitions_consumed": local,
    })
    return view


def validate_record(record, experiment, seed, condition, expected_episodes=256):
    """Reject stale, misrouted, or internally inconsistent held-out files."""
    expected = {
        "protocol": PROTOCOL,
        "env_id": experiment["env_id"],
        "method": experiment["method"],
        "condition": condition,
        "training_seed": seed,
        "checkpoint": "best.pt",
        "observation_contract": expected_observation_contract(experiment),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"held-out record {key} mismatch: expected {value!r}, "
                f"observed {record.get(key)!r}"
            )
    episodes = record.get("episode_records")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes:
        raise ValueError("held-out record has the wrong episode-record count")
    required_episode_fields = {
        "constraint_violated", "goals_completed", "intervention_occurred",
        *BRANCH_KEYS,
    }
    for episode_index, episode in enumerate(episodes):
        missing = required_episode_fields - episode.keys()
        if missing:
            raise ValueError(
                f"held-out episode {episode_index} lacks fields: {sorted(missing)}"
            )
        succeeded(episode)
    intervention_values = [episode["intervention_occurred"] for episode in episodes]
    if condition == "nominal" and any(value >= 0.5 for value in intervention_values):
        raise ValueError("nominal held-out record contains a physical intervention")
    if condition == "intervention" and any(value < 0.5 for value in intervention_values):
        raise ValueError("forced-intervention record contains an untriggered episode")
    if int(record.get("episodes", -1)) != len(episodes):
        raise ValueError("held-out episode count disagrees with episode records")
    validate_record_batch_seeds(record, expected_episodes)
    successes = sum(succeeded(episode) for episode in episodes)
    safe_successes = sum(
        succeeded(episode) and episode.get("constraint_violated", 0.0) < 0.5
        for episode in episodes
    )
    if int(record.get("successes", -1)) != successes:
        raise ValueError("held-out success count does not match episode records")
    if int(record.get("safe_successes", -1)) != safe_successes:
        raise ValueError("held-out safe-success count does not match episode records")
    if not math.isclose(float(record.get("success_rate", -1)), successes / len(episodes)):
        raise ValueError("held-out success rate does not match episode records")
    if not math.isclose(
        float(record.get("safe_success_rate", -1)), safe_successes / len(episodes),
    ):
        raise ValueError("held-out safe-success rate does not match episode records")
    record = transition_accounting_view(record, experiment)
    online_ppo_steps = int(record.get("online_ppo_environment_steps", -1))
    if online_ppo_steps != int(record.get("checkpoint_global_step", -2)):
        raise ValueError("held-out online PPO budget disagrees with checkpoint step")
    online_protocol_ppo = int(record.get("online_protocol_ppo_environment_steps", -1))
    if online_protocol_ppo != configured_ppo_budget(experiment):
        raise ValueError("online protocol PPO budget disagrees with configuration")
    if online_ppo_steps > online_protocol_ppo:
        raise ValueError("selected checkpoint exceeds completed training budget")
    initialization = record.get("initialization_provenance")
    initialization_ppo_steps = int(
        record.get("initialization_ppo_environment_steps", -1)
    )
    initialization_bc = int(
        record.get("initialization_bc_dagger_environment_transitions", -1)
    )
    initialization_protocol_ppo = int(
        record.get("initialization_protocol_ppo_environment_steps", -1)
    )
    configured_initialization = experiment.get("init_checkpoint")
    if configured_initialization:
        if not isinstance(initialization, dict):
            raise ValueError("initialized policy lacks initialization provenance")
        expected_checkpoint = str(configured_initialization).format(seed=seed)
        if initialization.get("checkpoint") != expected_checkpoint:
            raise ValueError("initializer checkpoint disagrees with configuration")
        if int(initialization.get("source_global_step", -1)) != initialization_ppo_steps:
            raise ValueError("initializer PPO budget is inconsistent")
        if int(initialization.get(
            "source_bc_dagger_environment_transitions", -1,
        )) != initialization_bc:
            raise ValueError("initializer DAgger budget is inconsistent")
        source_task = initialization.get("source_task", {})
        if initialization_protocol_ppo != configured_ppo_budget(source_task):
            raise ValueError("initializer protocol PPO budget disagrees with source task")
        if int(initialization.get(
            "source_protocol_ppo_environment_steps", -1,
        )) != initialization_protocol_ppo:
            raise ValueError("initializer completed-training marker is inconsistent")
        if initialization_ppo_steps > initialization_protocol_ppo:
            raise ValueError("initializer checkpoint exceeds its completed training budget")
        expected_source_bc = int(source_task.get("bc_pretrain_updates", 0)) * int(
            source_task.get("num_envs", 0)
        )
        if initialization_bc != expected_source_bc:
            raise ValueError("initializer DAgger budget disagrees with source task")
        if not initialization.get("source_observation_contract"):
            raise ValueError("initializer lacks observation-contract provenance")
        source_sha = initialization.get("source_sha256")
        if not isinstance(source_sha, dict) or not source_sha.get("trainer"):
            raise ValueError("initializer lacks source-code provenance")
    elif (
        initialization is not None or initialization_ppo_steps
        or initialization_protocol_ppo or initialization_bc
    ):
        raise ValueError("non-initialized policy reports initializer compute")
    ppo_steps = int(record.get("ppo_environment_steps", -1))
    if ppo_steps != online_ppo_steps + initialization_ppo_steps:
        raise ValueError("held-out total PPO budget omits initializer compute")
    local_bc = int(record.get("local_bc_dagger_environment_transitions", -1))
    bc_transitions = int(record.get("bc_dagger_environment_transitions", -1))
    expected_bc = int(experiment.get("bc_pretrain_updates", 0)) * int(
        experiment.get("num_envs", 0)
    )
    if local_bc != expected_bc:
        raise ValueError("held-out local DAgger budget disagrees with configuration")
    if bc_transitions != local_bc + initialization_bc:
        raise ValueError("held-out total DAgger budget omits initializer compute")
    if int(record.get("total_environment_transitions", -1)) != ppo_steps + bc_transitions:
        raise ValueError("held-out total transition budget is inconsistent")
    protocol_ppo = int(record.get("protocol_ppo_environment_steps", -1))
    if protocol_ppo != online_protocol_ppo + initialization_protocol_ppo:
        raise ValueError("held-out protocol PPO budget omits selection compute")
    if int(record.get(
        "protocol_environment_transitions_consumed", -1,
    )) != protocol_ppo + bc_transitions:
        raise ValueError("held-out consumed transition budget is inconsistent")
    source = record.get("training_source_sha256")
    if not isinstance(source, dict) or not source.get("trainer") or not source.get("environment"):
        raise ValueError("held-out record lacks training-source provenance")
    evaluation_source = record.get("evaluation_source_sha256")
    if not isinstance(evaluation_source, dict) or not evaluation_source.get("evaluator"):
        raise ValueError("held-out record lacks evaluator-source provenance")
    progress_counts = record.get("visual_progress_counts")
    expects_progress = bool(experiment.get("actor_learned_goal_progress", False))
    if expects_progress and not isinstance(progress_counts, dict):
        raise ValueError("learned-progress policy lacks held-out confusion counts")
    if not expects_progress and progress_counts is not None:
        raise ValueError("policy without a progress head reports progress counts")
    if progress_counts is not None:
        required_counts = {
            "correct_bits", "total_bits", "correct_vectors", "total_vectors",
            "true_positive", "true_negative", "false_positive", "false_negative",
        }
        if set(progress_counts) != required_counts:
            raise ValueError("held-out progress confusion-count schema mismatch")
        counts = {key: int(progress_counts[key]) for key in required_counts}
        if any(value < 0 for value in counts.values()):
            raise ValueError("held-out progress counts must be nonnegative")
        if counts["total_bits"] <= 0 or counts["total_vectors"] <= 0:
            raise ValueError("held-out progress counts must be nonempty")
        confusion_total = sum(counts[key] for key in (
            "true_positive", "true_negative", "false_positive", "false_negative",
        ))
        if confusion_total != counts["total_bits"]:
            raise ValueError("held-out progress confusion counts do not sum to total bits")
        if counts["correct_bits"] != counts["true_positive"] + counts["true_negative"]:
            raise ValueError("held-out progress correct-bit count is inconsistent")
        if counts["correct_vectors"] > counts["total_vectors"]:
            raise ValueError("held-out progress vector counts are inconsistent")
        positives = counts["true_positive"] + counts["false_negative"]
        negatives = counts["true_negative"] + counts["false_positive"]
        expected_rates = {
            "visual_progress_bit_accuracy": counts["correct_bits"] / counts["total_bits"],
            "visual_progress_exact_accuracy": counts["correct_vectors"] / counts["total_vectors"],
            "visual_progress_positive_recall": (
                counts["true_positive"] / positives if positives else None
            ),
            "visual_progress_negative_recall": (
                counts["true_negative"] / negatives if negatives else None
            ),
            "visual_progress_target_positive_rate": positives / counts["total_bits"],
            "visual_progress_predicted_positive_rate": (
                counts["true_positive"] + counts["false_positive"]
            ) / counts["total_bits"],
        }
        expected_rates["visual_progress_balanced_accuracy"] = (
            0.5 * (
                expected_rates["visual_progress_positive_recall"]
                + expected_rates["visual_progress_negative_recall"]
            )
            if positives and negatives else None
        )
        for key, value in expected_rates.items():
            observed = record.get(key, "missing")
            consistent = (
                observed is None if value is None
                else isinstance(observed, (int, float))
                and math.isclose(float(observed), value)
            )
            if not consistent:
                raise ValueError(f"held-out {key} is inconsistent with confusion counts")
    return record


def hierarchical_binary_interval(groups, predicate, rng, repetitions=20000):
    """Resample trained seeds, then held-out Bernoulli episodes within seed."""
    counts = np.asarray([sum(predicate(item) for item in group) for group in groups])
    trials = np.asarray([len(group) for group in groups])
    if not len(groups) or np.any(trials == 0):
        raise ValueError("hierarchical bootstrap requires non-empty seed groups")
    probabilities = counts / trials
    seed_indices = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    selected_trials = trials[seed_indices]
    sampled_counts = rng.binomial(selected_trials, probabilities[seed_indices])
    samples = (sampled_counts / selected_trials).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _paired_hierarchical_samples(groups, rng, repetitions=20000):
    """Hierarchical bootstrap for per-episode paired outcomes in {-1, 0, 1}."""
    trials = np.asarray([len(group) for group in groups])
    if not len(groups) or np.any(trials == 0):
        raise ValueError("paired bootstrap requires non-empty seed groups")
    negative = np.asarray([sum(value < 0 for value in group) for group in groups]) / trials
    positive = np.asarray([sum(value > 0 for value in group) for group in groups]) / trials
    seed_indices = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    selected_trials = trials[seed_indices]
    # A multinomial draw over {-1, 0, +1} can be sampled as a binomial for
    # positives followed by a conditional binomial for negatives.
    sampled_positive = rng.binomial(selected_trials, positive[seed_indices])
    remaining = selected_trials - sampled_positive
    negative_conditional = negative / np.maximum(1.0 - positive, 1e-12)
    sampled_negative = rng.binomial(
        remaining, np.clip(negative_conditional[seed_indices], 0.0, 1.0),
    )
    return ((sampled_positive - sampled_negative) / selected_trials).mean(axis=1)


def paired_effect(left_groups, right_groups, rng):
    if len(left_groups) != len(right_groups):
        raise ValueError("paired methods have unequal training-seed counts")
    raw_groups, safe_groups = [], []
    for left, right in zip(left_groups, right_groups):
        if len(left) != len(right):
            raise ValueError("paired methods have unequal per-seed episode counts")
        for episode_index, (a, b) in enumerate(zip(left, right)):
            for key in BRANCH_KEYS:
                if key in a or key in b:
                    if a.get(key) != b.get(key):
                        raise ValueError(
                            f"paired branch mismatch at episode {episode_index}: {key}"
                        )
        raw_groups.append([
            float(succeeded(a)) - float(succeeded(b)) for a, b in zip(left, right)
        ])
        safe_groups.append([
            float(succeeded(a) and a.get("constraint_violated", 0.0) < 0.5)
            - float(succeeded(b) and b.get("constraint_violated", 0.0) < 0.5)
            for a, b in zip(left, right)
        ])
    raw = np.concatenate(raw_groups)
    safe = np.concatenate(safe_groups)
    raw_samples = _paired_hierarchical_samples(raw_groups, rng)
    safe_samples = _paired_hierarchical_samples(safe_groups, rng)
    return {
        "paired_training_seeds": len(raw_groups), "paired_episodes": len(raw),
        "success_rate_difference": float(raw.mean()),
        "paired_bootstrap_95": [float(np.quantile(raw_samples, 0.025)), float(np.quantile(raw_samples, 0.975))],
        "safe_success_rate_difference": float(safe.mean()),
        "safe_paired_bootstrap_95": [float(np.quantile(safe_samples, 0.025)), float(np.quantile(safe_samples, 0.975))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/visual_recovery_ppo_gate_v1.json")
    parser.add_argument("--output", default="results/visual_recovery_ppo")
    parser.add_argument(
        "--conditions", nargs="+", choices=("nominal", "intervention"),
        default=("nominal", "intervention"),
    )
    parser.add_argument("--filename", default="aggregate.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    env_ids = {experiment["env_id"] for experiment in config["experiments"]}
    semantics = {
        "event_reward_intervention_target_only_v3"
        if env_id == "LearnedRecovery-v3"
        else "intervention_target_only_v2"
        for env_id in env_ids
    }
    if len(semantics) != 1:
        raise ValueError("one aggregate cannot mix benchmark semantics")
    expected_semantics = semantics.pop()
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "protocol": PROTOCOL,
        "benchmark_semantics": expected_semantics,
        "conditions": {},
    }
    rng = np.random.default_rng(20260827)
    for condition in args.conditions:
        filename = f"heldout_eval_{condition}.json"
        by_method = {}
        for experiment in config["experiments"]:
            method = experiment["method"]
            records = []
            for seed in config["seeds"]:
                path = root / method / f"seed_{seed}" / filename
                if not path.exists():
                    raise FileNotFoundError(path)
                records.append(validate_record(
                    json.loads(path.read_text(encoding="utf-8")),
                    experiment, seed, condition,
                    int(config.get("heldout_eval_episodes_per_seed", 256)),
                ))
            if any(
                record.get("benchmark_semantics") != expected_semantics
                for record in records
            ):
                raise ValueError("held-out records do not match configured benchmark semantics")
            episodes = [episode for record in records for episode in record["episode_records"]]
            successes = sum(succeeded(episode) for episode in episodes)
            safe = sum(succeeded(episode) and episode.get("constraint_violated", 0.0) < 0.5 for episode in episodes)
            by_method[method] = {
                "method": method, "seeds": len(records), "episodes": len(episodes),
                "successes": successes, "success_rate": successes / len(episodes),
                "success_wilson_95": wilson(successes, len(episodes)),
                "success_hierarchical_bootstrap_95": hierarchical_binary_interval(
                    [record["episode_records"] for record in records], succeeded, rng,
                ),
                "safe_successes": safe, "safe_success_rate": safe / len(episodes),
                "safe_success_wilson_95": wilson(safe, len(episodes)),
                "safe_success_hierarchical_bootstrap_95": hierarchical_binary_interval(
                    [record["episode_records"] for record in records],
                    lambda episode: succeeded(episode)
                    and episode.get("constraint_violated", 0.0) < 0.5,
                    rng,
                ),
                "constraint_violation_rate": float(np.mean([episode.get("constraint_violated", 0.0) for episode in episodes])),
                "mean_goals_completed": float(np.mean([episode.get("goals_completed", 0.0) for episode in episodes])),
                "seed_success_rates": [record["success_rate"] for record in records],
                "checkpoint_global_steps": [record["checkpoint_global_step"] for record in records],
                "online_ppo_environment_steps": [
                    record["online_ppo_environment_steps"] for record in records
                ],
                "initialization_ppo_environment_steps": [
                    record["initialization_ppo_environment_steps"] for record in records
                ],
                "ppo_environment_steps": [record["ppo_environment_steps"] for record in records],
                "online_protocol_ppo_environment_steps": [
                    record["online_protocol_ppo_environment_steps"] for record in records
                ],
                "initialization_protocol_ppo_environment_steps": [
                    record["initialization_protocol_ppo_environment_steps"]
                    for record in records
                ],
                "protocol_ppo_environment_steps": [
                    record["protocol_ppo_environment_steps"] for record in records
                ],
                "bc_dagger_environment_transitions": [
                    record["bc_dagger_environment_transitions"] for record in records
                ],
                "local_bc_dagger_environment_transitions": [
                    record["local_bc_dagger_environment_transitions"]
                    for record in records
                ],
                "initialization_bc_dagger_environment_transitions": [
                    record["initialization_bc_dagger_environment_transitions"]
                    for record in records
                ],
                "total_environment_transitions": [
                    record["total_environment_transitions"] for record in records
                ],
                "protocol_environment_transitions_consumed": [
                    record["protocol_environment_transitions_consumed"]
                    for record in records
                ],
                "seed_base": records[0]["seed_base"],
                "seed_results": records,
            }
            if len({record["seed_base"] for record in records}) != 1:
                raise ValueError("visual seed results use inconsistent held-out seed bases")
            if len({
                json.dumps(record["evaluation_source_sha256"], sort_keys=True)
                for record in records
            }) != 1:
                raise ValueError("visual seed results use inconsistent evaluator provenance")
            progress_counts = [record.get("visual_progress_counts") for record in records]
            if all(progress_counts):
                correct_bits = sum(item["correct_bits"] for item in progress_counts)
                total_bits = sum(item["total_bits"] for item in progress_counts)
                correct_vectors = sum(item["correct_vectors"] for item in progress_counts)
                total_vectors = sum(item["total_vectors"] for item in progress_counts)
                true_positive = sum(item["true_positive"] for item in progress_counts)
                true_negative = sum(item["true_negative"] for item in progress_counts)
                false_positive = sum(item["false_positive"] for item in progress_counts)
                false_negative = sum(item["false_negative"] for item in progress_counts)
                positives = true_positive + false_negative
                negatives = true_negative + false_positive
                by_method[method]["visual_progress_bit_accuracy"] = correct_bits / total_bits
                by_method[method]["visual_progress_exact_accuracy"] = (
                    correct_vectors / total_vectors
                )
                by_method[method]["visual_progress_positive_recall"] = (
                    true_positive / positives if positives else None
                )
                by_method[method]["visual_progress_negative_recall"] = (
                    true_negative / negatives if negatives else None
                )
                by_method[method]["visual_progress_balanced_accuracy"] = (
                    0.5 * (true_positive / positives + true_negative / negatives)
                    if positives and negatives else None
                )
                by_method[method]["visual_progress_target_positive_rate"] = (
                    positives / total_bits
                )
                by_method[method]["visual_progress_predicted_positive_rate"] = (
                    (true_positive + false_positive) / total_bits
                )
                by_method[method]["visual_progress_counts"] = {
                    "correct_bits": correct_bits,
                    "total_bits": total_bits,
                    "correct_vectors": correct_vectors,
                    "total_vectors": total_vectors,
                    "true_positive": true_positive,
                    "true_negative": true_negative,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                }
                by_method[method]["seed_visual_progress_bit_accuracy"] = [
                    record["visual_progress_bit_accuracy"] for record in records
                ]
                by_method[method]["seed_visual_progress_balanced_accuracy"] = [
                    record["visual_progress_balanced_accuracy"] for record in records
                ]
            by_method[method]["_episodes"] = episodes
            by_method[method]["_seed_episodes"] = [
                record["episode_records"] for record in records
            ]
        comparisons = []
        for left, right in itertools.combinations(by_method, 2):
            result = {"left": left, "right": right}
            result.update(paired_effect(
                by_method[left]["_seed_episodes"],
                by_method[right]["_seed_episodes"], rng,
            ))
            comparisons.append(result)
        for result in by_method.values():
            result.pop("_episodes")
            result.pop("_seed_episodes")
        payload["conditions"][condition] = {
            "methods": list(by_method.values()), "paired_comparisons": comparisons,
        }
    path = root / args.filename
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
