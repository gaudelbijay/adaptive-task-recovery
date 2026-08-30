import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_visual_task_representation_probes import (  # noqa: E402
    AGGREGATE_PROTOCOL,
)
from compare_visual_task_representations import compare, markdown  # noqa: E402
from probe_visual_task_representation import binary_auc, binary_probe  # noqa: E402


def test_binary_task_probe_reports_imbalance_robust_metrics():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    assert binary_auc(torch.tensor([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert binary_auc(torch.tensor([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert binary_auc(torch.ones(4), labels) == 0.5

    torch.manual_seed(3)
    train_y = torch.randint(0, 2, (512, 2), dtype=torch.double)
    test_y = torch.randint(0, 2, (512, 2), dtype=torch.double)
    train_x = torch.cat((train_y, torch.randn(512, 6, dtype=torch.double)), dim=1)
    test_x = torch.cat((test_y, torch.randn(512, 6, dtype=torch.double)), dim=1)
    result = binary_probe(train_x, train_y, test_x, test_y, 1.0)
    assert result["macro_balanced_accuracy"] > 0.99
    assert result["macro_roc_auc"] > 0.99
    assert all(0 < value < 1 for value in result["per_target_positive_prevalence"])


def _aggregate(path, method, learned_offset=0.0):
    records = []
    for seed in (1788, 4796, 9351):
        learned = {
            "r2_variance_weighted": 0.4 + learned_offset,
            "macro_balanced_accuracy": 0.8 + learned_offset,
            "macro_roc_auc": 0.9 + learned_offset,
        }
        random = {
            "r2_variance_weighted": 0.1,
            "macro_balanced_accuracy": 0.55,
            "macro_roc_auc": 0.60,
        }
        records.append({
            "training_seed": seed,
            "train_samples": 8192, "test_samples": 8192,
            "ridge_regularization": 1.0,
            "targets": ["red_goal_resolved", "blue_goal_resolved"],
            "probe_source_sha256": {"probe": "same", "environment_registration": "same"},
            "learned_encoder": learned, "random_encoder": random,
            "learned_minus_random_r2": learned["r2_variance_weighted"] - random["r2_variance_weighted"],
            "learned_minus_random_balanced_accuracy": learned["macro_balanced_accuracy"] - random["macro_balanced_accuracy"],
            "learned_minus_random_roc_auc": learned["macro_roc_auc"] - random["macro_roc_auc"],
            "probe_dataset": {
                "behavior_checkpoint": f"behavior/seed_{seed}/best.pt",
                "behavior_method": "fixed", "behavior_checkpoint_global_step": 10,
                "behavior_observation_contract": "visual",
                "train_seed": 93000000 + seed * 10,
                "test_seed": 93000000 + seed * 10 + 1,
                "train_sha256": f"{seed:064x}",
                "test_sha256": f"{seed + 1:064x}",
            },
        })
    path.write_text(json.dumps({
        "protocol": AGGREGATE_PROTOCOL,
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "methods": [{"method": method, "training_seeds": 3, "seed_results": records}],
    }))


def test_task_representation_comparison_requires_matched_pixels(tmp_path):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    _aggregate(control, "control")
    _aggregate(treatment, "treatment", 0.05)
    config = {
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "required_training_seeds": 3,
        "methods": [
            {"name": "control", "method": "control", "path": str(control)},
            {"name": "treatment", "method": "treatment", "path": str(treatment)},
        ],
        "primary_diagnostic": {"treatment": "treatment", "control": "control"},
    }
    result = compare(config)
    assert result["dataset_match_verified"] is True
    assert np.isclose(
        result["primary_diagnostic"]["paired_metrics"]["balanced_accuracy"][
            "mean_difference"
        ], 0.05,
    )

    invalid = json.loads(treatment.read_text())
    invalid["methods"][0]["seed_results"][0]["probe_dataset"][
        "train_sha256"
    ] = "f" * 64
    treatment.write_text(json.dumps(invalid))
    try:
        compare(config)
    except ValueError as error:
        assert "pixels differ" in str(error)
    else:
        raise AssertionError("mismatched semantic-probe pixels were accepted")


def test_task_representation_markdown_names_the_configured_pair(tmp_path):
    control = tmp_path / "control.json"
    treatment = tmp_path / "treatment.json"
    _aggregate(control, "control")
    _aggregate(treatment, "treatment", 0.05)
    payload = compare({
        "benchmark_semantics": "event_reward_intervention_target_only_v3",
        "required_training_seeds": 3,
        "methods": [
            {"name": "control_encoder", "method": "control", "path": str(control)},
            {
                "name": "low_variance_encoder", "method": "treatment",
                "path": str(treatment),
            },
        ],
        "primary_diagnostic": {
            "treatment": "low_variance_encoder", "control": "control_encoder",
        },
    })
    report = markdown(payload)
    assert "Paired low_variance_encoder − control_encoder diagnostic" in report
    assert "V16 − V15" not in report


def test_task_probe_is_separate_from_frozen_pose_probe():
    task_source = (
        ROOT / "scripts/probe_visual_task_representation.py"
    ).read_text(encoding="utf-8")
    pose_source = (
        ROOT / "scripts/probe_visual_representation.py"
    ).read_text(encoding="utf-8")
    assert '"critic_goal_resolved"' in task_source
    assert '"task_representation_probe.json"' in task_source
    assert '"representation_probe.json"' in pose_source
    assert '"task_representation_probe.json"' not in pose_source


def test_pose_probe_supports_non_overwriting_frozen_environment_protocol():
    pose_source = (
        ROOT / "scripts/probe_visual_representation.py"
    ).read_text(encoding="utf-8")
    probe_wrapper = (
        ROOT / "scripts/slurm_visual_representation_probe.sh"
    ).read_text(encoding="utf-8")
    aggregate_wrapper = (
        ROOT / "scripts/slurm_visual_representation_aggregate.sh"
    ).read_text(encoding="utf-8")
    protocol = json.loads((
        ROOT / "configs/visual_representation_probe_strict_matched_v1.json"
    ).read_text(encoding="utf-8"))
    assert '"--probe-protocol-config"' in pose_source
    assert '"--filename"' in pose_source
    assert 'run_dir / args.filename' in pose_source
    assert "ATR_PROBE_PROTOCOL_CONFIG" in probe_wrapper
    assert "ATR_REPRESENTATION_PROBE_FILENAME" in probe_wrapper
    assert "--probe-filename" in aggregate_wrapper
    assert protocol["env_kwargs"]["intervention_probability"] == 1.0
    assert protocol["env_kwargs"]["onset_step_range"] == [0, 0]
    assert protocol["env_kwargs"]["intervention_force"] == 6.0
    assert protocol["env_kwargs"]["intervention_steps"] == 24
