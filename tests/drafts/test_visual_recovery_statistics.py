"""Regression tests for seed-aware visual-policy uncertainty estimates."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_visual_recovery import (  # noqa: E402
    configured_ppo_budget,
    hierarchical_binary_interval,
    paired_effect,
    succeeded,
    validate_record,
)
from compare_visual_recovery_candidates import (  # noqa: E402
    classify_missing_candidates,
    comparison_markdown,
    paired_seed_groups,
    visual_method,
)
from aggregate_manipulation_results import (  # noqa: E402
    hierarchical_binary_interval as state_hierarchical_binary_interval,
)
from plot_visual_recovery_learning import method_label  # noqa: E402
from plot_visual_recovery_results import (  # noqa: E402
    interval_errors,
    validate as validate_result_figure,
)
from aggregate_visual_representation_probes import (  # noqa: E402
    TARGETS,
    seed_bootstrap_interval,
    validate_probe_record,
)
from check_visual_competence_gate import (  # noqa: E402
    check_gate,
    check_visualization_gate,
)
from validate_visual_recovery_hypotheses import validate  # noqa: E402
from compare_visual_representations import compare as compare_representations  # noqa: E402
from check_visual_confirmatory_gate import confirmatory_gate  # noqa: E402
from aggregate_five_seed_visual_confirmation import combine as combine_five_seed  # noqa: E402
from plot_five_seed_visual_confirmation import validate as validate_five_seed_figure  # noqa: E402
from evaluation_seed import (  # noqa: E402
    MAX_LEGACY_RANDOMSTATE_SEED,
    SEED_DERIVATION,
    heldout_batch_seed,
    validate_record_batch_seeds,
)


def _episodes(success, count=64):
    return [
        {"success_once": float(success), "constraint_violated": 0.0}
        for _ in range(count)
    ]


def test_heldout_seed_derivation_preserves_screen_and_handles_confirmation():
    seed_base = 81_000_000
    # Preserve the frozen screening protocol exactly.
    assert heldout_batch_seed(seed_base, 9351, 0) == seed_base + 9351 * 100_000
    assert heldout_batch_seed(seed_base, 1788, 224) == (
        seed_base + 1788 * 100_000 + 224
    )

    # Confirmation seeds overflow ManiSkill's legacy RandomState constructor
    # under the linear formula. The fallback remains deterministic, in range,
    # and collision-free across every held-out vector batch.
    values = [
        heldout_batch_seed(seed_base, seed, completed)
        for seed in (71064, 84293)
        for completed in range(0, 256, 32)
    ]
    assert values == [
        heldout_batch_seed(seed_base, seed, completed)
        for seed in (71064, 84293)
        for completed in range(0, 256, 32)
    ]
    assert len(values) == len(set(values))
    assert all(0 <= value <= MAX_LEGACY_RANDOMSTATE_SEED for value in values)
    record = {
        "seed_base": seed_base,
        "training_seed": 71064,
        "seed_derivation": SEED_DERIVATION,
        "batch_seeds": values[:8],
    }
    validate_record_batch_seeds(record, 256)


def test_overflowing_heldout_seed_fails_closed_without_provenance():
    record = {"seed_base": 81_000_000, "training_seed": 84293}
    try:
        validate_record_batch_seeds(record, 256)
    except ValueError as error:
        assert "lacks 31-bit seed provenance" in str(error)
    else:
        raise AssertionError("overflowing legacy seed unexpectedly validated")


def test_hierarchical_interval_retains_between_policy_seed_uncertainty():
    groups = [_episodes(True), _episodes(False), _episodes(False)]
    interval = hierarchical_binary_interval(
        groups, succeeded, np.random.default_rng(7), repetitions=20000,
    )
    # A flat 192-episode binomial interval would be spuriously narrow. With
    # only three independently trained policies, resampling seeds correctly
    # retains both extreme outcomes in the 95% interval.
    assert interval == [0.0, 1.0]


def test_configured_visual_ppo_budget_matches_complete_rollout_batches():
    assert configured_ppo_budget({
        "num_envs": 2, "num_steps": 5, "total_timesteps": 95,
    }) == 90


def test_state_aggregation_uses_the_same_seed_hierarchy():
    groups = [_episodes(True), _episodes(False), _episodes(False)]
    interval = state_hierarchical_binary_interval(
        groups, succeeded, np.random.default_rng(7), repetitions=20000,
    )
    assert interval == [0.0, 1.0]


def test_learning_curve_label_discloses_training_only_components():
    label = method_label({
        "bc_teacher_checkpoint": "teacher.pt",
        "asymmetric_critic": True,
        "privileged_aux_coefficient": 0.1,
        "temporal_ssl_coefficient": 0.05,
        "actor_learned_goal_progress": True,
    })
    assert label == (
        "DAgger + asymmetric critic + pose auxiliary + temporal SSL + progress head"
    )


def test_final_result_figure_requires_complete_v3_heldout_input():
    row = {
        "name": "direct_rgb_ppo", "nominal_success_rate": 0.5,
        "nominal_success_hierarchical_bootstrap_95": [0.4, 0.6],
        "safe_success_rate": 0.3,
        "safe_success_hierarchical_bootstrap_95": [0.2, 0.4],
        "constraint_violation_rate": 0.01,
    }
    payload = {
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "required_missing": [], "candidates": [row],
    }
    assert validate_result_figure(payload) == [row]
    assert np.allclose(interval_errors(0.5, [0.4, 0.7]), [0.1, 0.2])
    payload["required_missing"] = ["missing.json"]
    try:
        validate_result_figure(payload)
    except ValueError as error:
        assert "missing required" in str(error)
    else:
        raise AssertionError("incomplete comparison was plotted")


def test_representation_probe_bootstrap_resamples_training_seeds():
    interval = seed_bootstrap_interval(
        [1.0, 0.0, 0.0], np.random.default_rng(7), repetitions=20000
    )
    assert interval == [0.0, 1.0]


def test_representation_probe_rejects_wrong_semantics_or_metric_arithmetic():
    experiment = {
        "env_id": "LearnedRecovery-v3", "method": "temporal",
        "actor_tcp_pose": True,
    }
    record = {
        "protocol": "held-out linear pose probe; labels unavailable to actor",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "observation_contract": "rgb_qpos_qvel_tcp_instruction_v2",
        "method": "temporal", "training_seed": 9351,
        "train_samples": 8, "test_samples": 8,
        "ridge_regularization": 1.0, "targets": TARGETS,
        "checkpoint_global_step": 10,
        "training_source_sha256": {"trainer": "train"},
        "probe_source_sha256": {"probe": "probe"},
        "probe_dataset": {
            "protocol": "frozen seed-matched behavior policy; identical pixels across methods",
            "behavior_checkpoint": "behavior/seed_9351/best.pt",
            "behavior_method": "fixed_behavior",
            "behavior_checkpoint_global_step": 10,
            "behavior_observation_contract": "rgb_robot_proprio_instruction_visual_progress_v4",
            "behavior_training_source_sha256": {"trainer": "behavior"},
            "train_seed": 93093510,
            "test_seed": 93093511,
            "train_sha256": "a" * 64,
            "test_sha256": "b" * 64,
        },
        "learned_encoder": {"r2_variance_weighted": 0.4},
        "random_encoder": {"r2_variance_weighted": 0.1},
        "learned_minus_random_r2": 0.3,
    }
    behavior = "behavior/seed_{seed}/best.pt"
    assert validate_probe_record(record, experiment, 9351, 8, behavior, 1.0) is record
    invalid = copy.deepcopy(record)
    invalid["benchmark_semantics"] = "intervention_target_only_v2"
    try:
        validate_probe_record(invalid, experiment, 9351, 8, behavior, 1.0)
    except ValueError as error:
        assert "benchmark_semantics mismatch" in str(error)
    else:
        raise AssertionError("V2 probe was accepted for a V3 experiment")
    invalid = copy.deepcopy(record)
    invalid["learned_minus_random_r2"] = 0.5
    try:
        validate_probe_record(invalid, experiment, 9351, 8, behavior, 1.0)
    except ValueError as error:
        assert "metric is inconsistent" in str(error)
    else:
        raise AssertionError("arithmetically inconsistent probe was accepted")
    invalid = copy.deepcopy(record)
    invalid["probe_dataset"]["test_sha256"] = "different"
    try:
        validate_probe_record(invalid, experiment, 9351, 8, behavior, 1.0)
    except ValueError as error:
        assert "not a SHA-256" in str(error)
    else:
        raise AssertionError("malformed probe dataset digest was accepted")
    invalid = copy.deepcopy(record)
    invalid["probe_dataset"]["behavior_checkpoint"] = "wrong.pt"
    try:
        validate_probe_record(invalid, experiment, 9351, 8, behavior, 1.0)
    except ValueError as error:
        assert "behavior checkpoint disagrees" in str(error)
    else:
        raise AssertionError("misrouted probe behavior checkpoint was accepted")


def test_representation_comparison_requires_byte_identical_datasets(tmp_path):
    semantics = "event_reward_intervention_target_only_v3"
    seeds = [9351, 4796, 1788]

    def aggregate(path, method, learned):
        records = []
        for seed, value in zip(seeds, learned):
            random = 0.1
            records.append({
                "training_seed": seed,
                "train_samples": 8192, "test_samples": 8192,
                "ridge_regularization": 1.0, "targets": TARGETS,
                "probe_source_sha256": {"probe": "same", "environment_registration": "same"},
                "learned_encoder": {"r2_variance_weighted": value},
                "random_encoder": {"r2_variance_weighted": random},
                "learned_minus_random_r2": value - random,
                "probe_dataset": {
                    "behavior_checkpoint": f"behavior/seed_{seed}/best.pt",
                    "behavior_method": "fixed",
                    "behavior_checkpoint_global_step": 10,
                    "behavior_observation_contract": "visual",
                    "train_seed": 93000000 + seed * 10,
                    "test_seed": 93000000 + seed * 10 + 1,
                    "train_sha256": f"{seed:064x}",
                    "test_sha256": f"{seed + 1:064x}",
                },
            })
        payload = {
            "protocol": "held-out linear pose probe aggregated over training seeds",
            "benchmark_semantics": semantics,
            "methods": [{
                "method": method, "training_seeds": 3,
                "seed_results": records,
            }],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    control_path = tmp_path / "control.json"
    treatment_path = tmp_path / "treatment.json"
    aggregate(control_path, "control_method", [0.2, 0.3, 0.4])
    aggregate(treatment_path, "treatment_method", [0.4, 0.5, 0.6])
    config = {
        "benchmark_semantics": semantics, "required_training_seeds": 3,
        "methods": [
            {"name": "control", "method": "control_method", "path": str(control_path)},
            {"name": "treatment", "method": "treatment_method", "path": str(treatment_path)},
        ],
        "primary_diagnostic": {"treatment": "treatment", "control": "control"},
    }
    result = compare_representations(config)
    assert result["dataset_match_verified"] is True
    assert np.isclose(result["primary_diagnostic"]["mean_r2_difference"], 0.2)

    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["methods"][0]["seed_results"][0]["probe_dataset"][
        "train_sha256"
    ] = "f" * 64
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        compare_representations(config)
    except ValueError as error:
        assert "identical pixels" in str(error)
    else:
        raise AssertionError("mismatched representation pixels were accepted")
    aggregate(treatment_path, "treatment_method", [0.4, 0.5, 0.6])
    payload = json.loads(treatment_path.read_text(encoding="utf-8"))
    payload["methods"][0]["seed_results"][0]["ridge_regularization"] = 0.5
    treatment_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        compare_representations(config)
    except ValueError as error:
        assert "different probe protocols" in str(error)
    else:
        raise AssertionError("mismatched ridge probe was accepted")


def test_competence_gate_fails_closed_on_protocol_and_threshold():
    method = "visual"
    payload = {
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "conditions": {"nominal": {"methods": [{
            "method": method, "seeds": 3, "episodes": 768,
            "success_rate": 0.70,
            "success_hierarchical_bootstrap_95": [0.5, 0.9],
            "constraint_violation_rate": 0.0,
        }]}}
    }
    assert check_gate(payload, method, 0.70, 3, 768)["passed"] is True
    payload["conditions"]["nominal"]["methods"][0]["success_rate"] = 0.699
    assert check_gate(payload, method, 0.70, 3, 768)["passed"] is False
    payload["benchmark_semantics"] = "intervention_target_only_v2"
    try:
        check_gate(payload, method, 0.70, 3, 768)
    except ValueError as error:
        assert "V3 event-reward semantics" in str(error)
    else:
        raise AssertionError("V2 aggregate passed the V3 release gate")


def test_confirmatory_gate_uses_frozen_state_thresholds_and_new_seeds():
    method = "visual"
    state_name = "state"
    visual = {
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "conditions": {
            "nominal": {"methods": [{
                "method": method, "seeds": 3, "episodes": 768,
                "success_rate": 0.8, "safe_success_rate": 0.8,
                "constraint_violation_rate": 0.0,
            }]},
            "intervention": {"methods": [{
                "method": method, "seeds": 3, "episodes": 768,
                "success_rate": 0.6, "safe_success_rate": 0.59,
                "constraint_violation_rate": 0.01,
            }]},
        },
    }
    state = {
        "environments": [{
            "method": state_name, "seeds": 3, "episodes": 768,
            "pooled_success_rate": 0.55, "pooled_safe_success_rate": 0.54,
            "constraint_violation_rate": 0.02,
        }],
    }
    result = confirmatory_gate(
        visual, state, method, state_name, [71064, 84293],
    )
    assert result["passed"] is True
    assert result["no_confirmatory_seed_may_be_discarded"] is True
    visual["conditions"]["intervention"]["methods"][0][
        "safe_success_rate"
    ] = 0.53
    assert confirmatory_gate(
        visual, state, method, state_name, [71064, 84293],
    )["passed"] is False
    try:
        confirmatory_gate(visual, state, method, state_name, [71064, 9351])
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("screening seed leaked into confirmation")


def test_five_seed_combiner_keeps_every_predeclared_seed(tmp_path):
    semantics = "event_reward_intervention_target_only_v3"
    screen = [9351, 4796, 1788]
    confirm = [71064, 84293]
    visual_name = "visual"
    state_name = "state"

    def records(seeds, success, condition, visual=False):
        output = []
        for seed in seeds:
            episodes = [{
                "success_once": float(success),
                "constraint_violated": 0.0,
                "goals_completed": 2.0 if success else 0.0,
                "first_goal_removed": float(index % 2),
                "instruction_red_first": float((index + seed) % 2),
            } for index in range(256)]
            output.append({
                "training_seed": seed, "episode_records": episodes,
                "condition": condition, "seed_base": 81000000,
                **({
                    "training_source_sha256": {"trainer": "same"},
                    "evaluation_source_sha256": {"evaluator": "same"},
                } if visual else {}),
            })
        return output

    def visual_payload(seeds, success):
        nominal = {
            "method": visual_name, "seeds": len(seeds),
            "episodes": 256 * len(seeds),
            "seed_results": records(seeds, success, "nominal", visual=True),
        }
        intervention = copy.deepcopy(nominal)
        intervention["seed_results"] = records(
            seeds, success, "intervention", visual=True,
        )
        return {
            "benchmark_semantics": semantics,
            "conditions": {
                "nominal": {"methods": [nominal]},
                "intervention": {"methods": [intervention]},
            },
        }

    def state_payload(seeds, success):
        return {
            "benchmark_semantics": semantics,
            "environments": [{
                "method": state_name, "seeds": len(seeds),
                "episodes": 256 * len(seeds),
                "seed_results": records(seeds, success, "intervention"),
            }],
        }

    paths = {
        "visual_screening": tmp_path / "visual_screen.json",
        "visual_confirmatory": tmp_path / "visual_confirm.json",
        "state_screening": tmp_path / "state_screen.json",
        "state_confirmatory": tmp_path / "state_confirm.json",
    }
    payloads = {
        "visual_screening": visual_payload(screen, True),
        "visual_confirmatory": visual_payload(confirm, False),
        "state_screening": state_payload(screen, False),
        "state_confirmatory": state_payload(confirm, True),
    }
    for key, path in paths.items():
        path.write_text(json.dumps(payloads[key]), encoding="utf-8")
    config = {
        "benchmark_semantics": semantics,
        "screening_seeds": screen, "confirmatory_seeds": confirm,
        "visual_method": visual_name, "state_method": state_name,
        **{key: str(path) for key, path in paths.items()},
    }
    result = combine_five_seed(config)
    assert result["no_seed_discarded"] is True
    assert result["all_training_seeds"] == sorted(screen + confirm)
    assert result["visual"]["intervention"]["episodes"] == 1280
    assert result["paired_visual_minus_state"]["paired_training_seeds"] == 5

    payloads["visual_confirmatory"]["conditions"]["nominal"]["methods"][0][
        "seed_results"
    ] = payloads["visual_confirmatory"]["conditions"]["nominal"]["methods"][0][
        "seed_results"
    ][:-1]
    paths["visual_confirmatory"].write_text(
        json.dumps(payloads["visual_confirmatory"]), encoding="utf-8"
    )
    try:
        combine_five_seed(config)
    except ValueError as error:
        assert "missing, duplicate, or unexpected" in str(error)
    else:
        raise AssertionError("five-seed report silently discarded a seed")


def test_five_seed_figure_refuses_incomplete_confirmation():
    result = {
        "seeds": 5, "episodes": 1280,
        "success_rate": 0.8, "safe_success_rate": 0.79,
        "constraint_violation_rate": 0.01,
        "success_hierarchical_bootstrap_95": [0.7, 0.9],
        "safe_success_hierarchical_bootstrap_95": [0.69, 0.89],
    }
    payload = {
        "protocol": "five-seed confirmatory held-out visual/state comparison",
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "no_seed_discarded": True,
        "all_training_seeds": [1, 2, 3, 4, 5],
        "heldout_episodes_per_condition": 1280,
        "visual_training_source_sha256": {"trainer": "x"},
        "visual_evaluation_source_sha256": {"evaluator": "y"},
        "visual": {
            "nominal": copy.deepcopy(result),
            "intervention": copy.deepcopy(result),
        },
        "state_forced_intervention": copy.deepcopy(result),
    }
    assert validate_five_seed_figure(payload)[1]["episodes"] == 1280
    payload["all_training_seeds"].pop()
    try:
        validate_five_seed_figure(payload)
    except ValueError as error:
        assert "exactly five" in str(error)
    else:
        raise AssertionError("incomplete five-seed figure was rendered")


def test_readme_visualization_requires_recovery_safety_and_nominal_retention():
    method = "visual"
    payload = {
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "conditions": {
            "intervention": {"methods": [{
                "method": method, "seeds": 3, "episodes": 768,
                "success_rate": 425 / 768, "safe_success_rate": 424 / 768,
                "constraint_violation_rate": 12 / 768,
            }]},
            "nominal": {"methods": [{
                "method": method, "seeds": 3, "episodes": 768,
                "success_rate": 0.70,
            }]},
        },
    }
    result = check_visualization_gate(
        payload, method, 425 / 768, 424 / 768, 12 / 768, 0.70,
    )
    assert result["passed"] is True
    payload["conditions"]["nominal"]["methods"][0]["success_rate"] = 0.699
    assert check_visualization_gate(
        payload, method, 425 / 768, 424 / 768, 12 / 768, 0.70,
    )["passed"] is False
    payload["conditions"]["nominal"]["methods"][0]["success_rate"] = 0.70
    payload["conditions"]["intervention"]["methods"][0][
        "constraint_violation_rate"
    ] = 13 / 768
    assert check_visualization_gate(
        payload, method, 425 / 768, 424 / 768, 12 / 768, 0.70,
    )["passed"] is False


def test_paired_effect_bootstraps_training_seeds_and_matched_episodes():
    left = [_episodes(True), _episodes(False), _episodes(False)]
    right = [_episodes(False), _episodes(False), _episodes(False)]
    result = paired_effect(left, right, np.random.default_rng(11))
    assert result["paired_training_seeds"] == 3
    assert result["paired_episodes"] == 192
    assert result["success_rate_difference"] == 1 / 3
    assert result["paired_bootstrap_95"] == [0.0, 1.0]


def test_visual_aggregate_rejects_misrouted_or_inconsistent_seed_file():
    experiment = {
        "env_id": "LearnedRecovery-v3", "method": "visual",
        "actor_tcp_pose": True, "num_envs": 2, "num_steps": 5,
        "total_timesteps": 100, "bc_pretrain_updates": 3,
    }
    episodes = [{
        **episode,
        "goals_completed": 2.0,
        "intervention_occurred": 0.0,
        "first_goal_removed": float(index % 2),
        "instruction_red_first": float(index % 2),
    } for index, episode in enumerate(_episodes(True, count=2))]
    record = {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "env_id": "LearnedRecovery-v3", "method": "visual",
        "condition": "nominal", "training_seed": 9351,
        "seed_base": 81_000_000,
        "checkpoint": "best.pt",
        "observation_contract": "rgb_qpos_qvel_tcp_instruction_v2",
        "episodes": 2, "episode_records": episodes,
        "successes": 2, "success_rate": 1.0,
        "safe_successes": 2, "safe_success_rate": 1.0,
        "checkpoint_global_step": 100, "online_ppo_environment_steps": 100,
        "initialization_ppo_environment_steps": 0,
        "ppo_environment_steps": 100,
        "online_protocol_ppo_environment_steps": 100,
        "initialization_protocol_ppo_environment_steps": 0,
        "protocol_ppo_environment_steps": 100,
        "local_bc_dagger_environment_transitions": 6,
        "initialization_bc_dagger_environment_transitions": 0,
        "bc_dagger_environment_transitions": 6,
        "total_environment_transitions": 106,
        "protocol_environment_transitions_consumed": 106,
        "initialization_provenance": None,
        "training_source_sha256": {"trainer": "a", "environment": "b"},
        "evaluation_source_sha256": {"evaluator": "c"},
    }
    assert validate_record(record, experiment, 9351, "nominal", 2) is record
    record["method"] = "wrong"
    try:
        validate_record(record, experiment, 9351, "nominal", 2)
    except ValueError as error:
        assert "method mismatch" in str(error)
    else:
        raise AssertionError("misrouted held-out file was accepted")


def test_visual_aggregate_validates_progress_confusion_matrix():
    experiment = {
        "env_id": "LearnedRecovery-v3", "method": "visual_progress",
        "actor_tcp_pose": True, "actor_learned_goal_progress": True,
        "num_envs": 2, "num_steps": 5, "total_timesteps": 100,
    }
    episodes = [{
        **episode,
        "goals_completed": 2.0,
        "intervention_occurred": 0.0,
        "first_goal_removed": float(index % 2),
        "instruction_red_first": float(index % 2),
    } for index, episode in enumerate(_episodes(True, count=2))]
    counts = {
        "correct_bits": 14, "total_bits": 20,
        "correct_vectors": 6, "total_vectors": 10,
        "true_positive": 6, "true_negative": 8,
        "false_positive": 2, "false_negative": 4,
    }
    record = {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "env_id": "LearnedRecovery-v3", "method": "visual_progress",
        "condition": "nominal", "training_seed": 9351,
        "seed_base": 81_000_000,
        "checkpoint": "best.pt",
        "observation_contract": "rgb_robot_proprio_instruction_visual_progress_v4",
        "episodes": 2, "episode_records": episodes,
        "successes": 2, "success_rate": 1.0,
        "safe_successes": 2, "safe_success_rate": 1.0,
        "checkpoint_global_step": 100, "online_ppo_environment_steps": 100,
        "initialization_ppo_environment_steps": 0,
        "ppo_environment_steps": 100,
        "online_protocol_ppo_environment_steps": 100,
        "initialization_protocol_ppo_environment_steps": 0,
        "protocol_ppo_environment_steps": 100,
        "local_bc_dagger_environment_transitions": 0,
        "initialization_bc_dagger_environment_transitions": 0,
        "bc_dagger_environment_transitions": 0,
        "total_environment_transitions": 100,
        "protocol_environment_transitions_consumed": 100,
        "initialization_provenance": None,
        "training_source_sha256": {"trainer": "a", "environment": "b"},
        "evaluation_source_sha256": {"evaluator": "c"},
        "visual_progress_counts": counts,
        "visual_progress_bit_accuracy": 14 / 20,
        "visual_progress_exact_accuracy": 6 / 10,
        "visual_progress_positive_recall": 6 / 10,
        "visual_progress_negative_recall": 8 / 10,
        "visual_progress_balanced_accuracy": 0.7,
        "visual_progress_target_positive_rate": 10 / 20,
        "visual_progress_predicted_positive_rate": 8 / 20,
    }
    assert validate_record(record, experiment, 9351, "nominal", 2) is record
    invalid = copy.deepcopy(record)
    invalid["visual_progress_counts"]["false_negative"] = 3
    try:
        validate_record(invalid, experiment, 9351, "nominal", 2)
    except ValueError as error:
        assert "confusion counts do not sum" in str(error)
    else:
        raise AssertionError("inconsistent progress confusion matrix was accepted")


def test_visual_aggregate_counts_initializer_compute_for_adaptive_policy():
    experiment = {
        "env_id": "LearnedRecovery-v3", "method": "adaptive",
        "actor_tcp_pose": True, "num_envs": 2, "num_steps": 5,
        "total_timesteps": 200,
        "init_checkpoint": "clean/seed_{seed}/best.pt",
    }
    episodes = [{
        **episode,
        "goals_completed": 1.0,
        "intervention_occurred": 1.0,
        "first_goal_removed": float(index % 2),
        "instruction_red_first": float(index % 2),
    } for index, episode in enumerate(_episodes(True, count=2))]
    record = {
        "protocol": "held-out deterministic restricted-input visual-policy evaluation",
        "env_id": "LearnedRecovery-v3", "method": "adaptive",
        "condition": "intervention", "training_seed": 9351,
        "seed_base": 81_000_000,
        "checkpoint": "best.pt",
        "observation_contract": "rgb_qpos_qvel_tcp_instruction_v2",
        "episodes": 2, "episode_records": episodes,
        "successes": 2, "success_rate": 1.0,
        "safe_successes": 2, "safe_success_rate": 1.0,
        "checkpoint_global_step": 200, "online_ppo_environment_steps": 200,
        "initialization_ppo_environment_steps": 100,
        "ppo_environment_steps": 300,
        "online_protocol_ppo_environment_steps": 200,
        "initialization_protocol_ppo_environment_steps": 100,
        "protocol_ppo_environment_steps": 300,
        "local_bc_dagger_environment_transitions": 0,
        "initialization_bc_dagger_environment_transitions": 6,
        "bc_dagger_environment_transitions": 6,
        "total_environment_transitions": 306,
        "protocol_environment_transitions_consumed": 306,
        "initialization_provenance": {
            "checkpoint": "clean/seed_9351/best.pt",
            "source_global_step": 100,
            "source_protocol_ppo_environment_steps": 100,
            "source_bc_dagger_environment_transitions": 6,
            "source_observation_contract": "rgb_qpos_qvel_tcp_instruction_v2",
            "source_sha256": {"trainer": "old", "environment": "env"},
            "source_task": {
                "bc_pretrain_updates": 3, "num_envs": 2, "num_steps": 5,
                "total_timesteps": 100,
            },
        },
        "training_source_sha256": {"trainer": "new", "environment": "env"},
        "evaluation_source_sha256": {"evaluator": "eval"},
    }
    assert validate_record(record, experiment, 9351, "intervention", 2) is record
    invalid = copy.deepcopy(record)
    invalid["total_environment_transitions"] = 300
    try:
        validate_record(invalid, experiment, 9351, "intervention", 2)
    except ValueError as error:
        assert "total transition budget" in str(error)
    else:
        raise AssertionError("initializer compute was omitted without rejection")


def test_visual_paired_effect_rejects_branch_mismatch():
    left = [[{
        "success_once": 1.0, "constraint_violated": 0.0,
        "first_goal_removed": 1.0, "instruction_red_first": 0.0,
    }]]
    right = [[{
        "success_once": 1.0, "constraint_violated": 0.0,
        "first_goal_removed": 0.0, "instruction_red_first": 0.0,
    }]]
    try:
        paired_effect(left, right, np.random.default_rng(4))
    except ValueError as error:
        assert "paired branch mismatch" in str(error)
    else:
        raise AssertionError("branch-mismatched visual episodes were paired")


def test_cross_method_pairing_requires_identical_scene_branches():
    def result(seed, first_removed):
        return {
            "training_seed": seed,
            "seed_base": 81000000,
            "episode_records": [{
                "success_once": 1.0,
                "first_goal_removed": float(first_removed),
                "instruction_red_first": 1.0,
            }],
        }

    candidate = {"seed_results": [result(9351, True), result(1788, False)]}
    reference = {"seed_results": [result(1788, False), result(9351, True)]}
    left, right = paired_seed_groups(candidate, reference)
    assert len(left) == len(right) == 2

    reference["seed_results"][1] = result(9351, False)
    try:
        paired_seed_groups(candidate, reference)
    except ValueError as error:
        assert "branch mismatch" in str(error)
    else:
        raise AssertionError("mismatched held-out branches were accepted")


def test_final_comparison_requires_selector_for_multi_method_aggregate():
    payload = {"conditions": {"nominal": {"methods": [
        {"method": "a", "success_rate": 0.1},
        {"method": "b", "success_rate": 0.2},
    ]}}}
    assert visual_method(payload, "nominal", "b")["success_rate"] == 0.2
    try:
        visual_method(payload, "nominal")
    except ValueError as error:
        assert "explicit method selector" in str(error)
    else:
        raise AssertionError("ambiguous multi-method aggregate was accepted")


def test_final_comparison_distinguishes_optional_missing_extension(tmp_path):
    present = tmp_path / "present.json"
    present.write_text("{}", encoding="utf-8")
    required = tmp_path / "required.json"
    optional = tmp_path / "optional.json"
    missing, required_missing = classify_missing_candidates([
        {"path": str(present)},
        {"path": str(required)},
        {"path": str(optional), "required": False},
    ])
    assert missing == [str(required), str(optional)]
    assert required_missing == [str(required)]


def test_generated_comparison_markdown_uses_hierarchical_intervals_and_compute():
    payload = {
        "reference": {
            "raw_success_rate": 0.55, "safe_success_rate": 0.54,
            "constraint_violation_rate": 0.01,
        },
        "candidates": [{
            "name": "visual", "nominal_success_rate": 0.8,
            "nominal_success_hierarchical_bootstrap_95": [0.7, 0.9],
            "raw_success_rate": 0.6,
            "raw_success_hierarchical_bootstrap_95": [0.5, 0.7],
            "safe_success_rate": 0.58,
            "safe_success_hierarchical_bootstrap_95": [0.48, 0.68],
            "constraint_violation_rate": 0.02,
            "protocol_environment_transitions_consumed": [141_000_000] * 3,
        }],
        "missing": ["optional.json"],
    }
    rendered = comparison_markdown(payload)
    assert "80.00% [70.00%, 90.00%]" in rendered
    assert "141,000,000, 141,000,000, 141,000,000" in rendered
    assert "hierarchical 95% bootstrap" in rendered
    assert "Optional missing protocol extensions: optional.json" in rendered


def test_hypothesis_report_does_not_let_extension_rescue_primary(tmp_path):
    semantics = "event_reward_intervention_target_only_v3"
    seeds = [1, 2, 3]

    def method(method_name, successes, safe=None, violation=0.0):
        safe = successes if safe is None else safe
        records = []
        for seed, success, safe_success in zip(seeds, successes, safe):
            episode = {
                "success_once": float(success),
                "constraint_violated": float(success and not safe_success),
                "first_goal_removed": 1.0,
                "instruction_red_first": float(seed % 2),
            }
            records.append({
                "training_seed": seed, "seed_base": 81,
                "episode_records": [episode],
            })
        rate = sum(successes) / len(successes)
        safe_rate = sum(safe) / len(safe)
        return {
            "method": method_name, "seeds": 3, "episodes": 3,
            "success_rate": rate, "safe_success_rate": safe_rate,
            "constraint_violation_rate": violation,
            "success_hierarchical_bootstrap_95": [rate, rate],
            "seed_results": records,
        }

    def visual(path, methods):
        payload = {
            "benchmark_semantics": semantics,
            "conditions": {
                "nominal": {"methods": methods},
                "intervention": {"methods": methods},
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    main_path = tmp_path / "main.json"
    direct_path = tmp_path / "direct.json"
    dagger_path = tmp_path / "dagger.json"
    strict_path = tmp_path / "strict.json"
    visual(main_path, [method("main", [1, 1, 1])])
    visual(direct_path, [
        method("symmetric", [0, 0, 0]),
        method("asymmetric", [1, 1, 1]),
        # Primary temporal comparison loses despite the asymmetric baseline.
        method("temporal", [0, 0, 0]),
    ])
    visual(dagger_path, [
        method("dagger", [0, 0, 0]),
        # The extension succeeds, but must not rewrite primary V3.
        method("dagger_temporal", [1, 1, 1]),
    ])
    paired = {
        "paired_training_seeds": 3, "paired_episodes": 3,
        "success_rate_difference": 1.0,
        "paired_bootstrap_95": [1.0, 1.0],
        "safe_success_rate_difference": 1.0,
        "safe_paired_bootstrap_95": [1.0, 1.0],
    }
    strict_protocol = (
        "held-out deterministic strict-actual-removal policy evaluation"
    )
    strict_path.write_text(json.dumps({
        "benchmark_semantics": semantics,
        "protocol": strict_protocol,
        "cohorts": [
            {"label": "clean", "method": "clean", "training_seeds": seeds,
             "episodes": 3, "success_rate": 0.0, "safe_success_rate": 0.0,
             "constraint_violation_rate": 0.0},
            {"label": "adaptive", "method": "adaptive", "training_seeds": seeds,
             "episodes": 3, "success_rate": 1.0, "safe_success_rate": 1.0,
             "constraint_violation_rate": 0.0},
            {"label": "state", "method": "state", "training_seeds": seeds,
             "episodes": 3, "success_rate": 0.0, "safe_success_rate": 0.0,
             "constraint_violation_rate": 0.0},
        ],
        "paired_comparisons": [
            {"left": "adaptive", "right": "state", **paired},
        ],
        "paired_comparisons_by_branch": {
            "first_goal_physically_removed": [
                {"left": "adaptive", "right": "clean", **paired},
            ],
        },
    }), encoding="utf-8")
    config = {
        "benchmark_semantics": semantics,
        "required_training_seeds": 3, "required_episodes": 3,
        "strict_removal_protocol": strict_protocol,
        "v1": {"minimum_nominal_success": 0.7,
            "primary": {"path": str(direct_path), "method": "symmetric", "disclosure": "test"},
            "fallback_visual_competence": {"path": str(main_path), "method": "main", "disclosure": "fallback"},
        },
        "v2": {"path": str(direct_path), "primary": {
            "treatment": "asymmetric", "control": "symmetric",
        }},
        "v3": {"minimum_pooled_improvement": 0.05,
            "primary": {"path": str(direct_path), "treatment": "temporal", "control": "asymmetric"},
            "protocol_extension": {"path": str(dagger_path), "treatment": "dagger_temporal", "control": "dagger"},
        },
        "v4": {"primary": {
            "strict_path": str(strict_path), "adaptive_label": "adaptive",
            "clean_label": "clean", "branch": "first_goal_physically_removed",
        }},
        "v5": {
            "strict_path": str(strict_path), "reference_label": "state",
            "primary_label": "adaptive",
        },
    }
    result = validate(config)
    assert result["hypotheses"]["V1"]["verdict"] == "rejected"
    assert result["hypotheses"]["V1"]["fallback_visual_competence"][
        "verdict"
    ] == "confirmed"
    assert result["hypotheses"]["V2"]["verdict"] == "confirmed"
    assert result["hypotheses"]["V3"]["verdict"] == "rejected"
    comparisons = result["hypotheses"]["V3"]["comparisons"]
    assert comparisons[0]["verdict"] == "rejected"
    assert comparisons[1]["verdict"] == "confirmed"
    assert result["hypotheses"]["V5"]["verdict"] == "confirmed"
