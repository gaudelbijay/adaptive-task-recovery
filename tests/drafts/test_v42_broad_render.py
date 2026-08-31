import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v42_changes_only_the_declared_training_identity_and_renderer_exposure():
    source = load("configs/visual_recovery_v40_three_seed_view.json")["experiments"][0]
    task = load("configs/visual_recovery_v42_broad_render_smoke.json")["experiments"][0]
    shared = {
        key: value for key, value in task.items()
        if key not in {
            "method", "source_visual_checkpoint", "source_training_protocol",
            "training_protocol", "environment_extension", "fine_tune_updates",
            "total_timesteps", "magnitude_threshold", "paired_environment_profiles",
            "profile_sampling_cycle",
        }
    }
    expected = {
        key: value for key, value in source.items()
        if key not in {
            "method", "source_visual_checkpoint", "fine_tune_updates",
            "total_timesteps", "magnitude_threshold", "paired_environment_profiles",
            "profile_sampling_cycle",
        }
    }
    assert shared == expected
    assert task["magnitude_threshold"] == 0.015
    assert task["source_training_protocol"] == "backkey_targeted_dense_repair_v19"
    assert task["training_protocol"] == "broad_render_dense_repair_v19"
    assert set(task["paired_environment_profiles"]) == {
        "camera_right_back_4cm", "lighting_bright_side", "lighting_green_ambient",
    }
    assert task["total_timesteps"] == (
        task["fine_tune_updates"] * task["num_envs"]
        * (1 + len(task["paired_environment_profiles"]))
    )


def test_v42_development_and_untouched_domains_are_disjoint():
    development = load("configs/v42_smoke_development_ood_v1.json")
    untouched = load("configs/v42_confirmatory_unseen_visual_ood_v1.json")
    dev_names = {item["name"] for item in development["variants"]}
    untouched_names = {item["name"] for item in untouched["variants"]}
    assert dev_names & untouched_names == {"baseline", "progress_cyclic_shift"}
    assert development["seed_base"] != untouched["seed_base"]
    assert untouched["seed_base"] == 127000000


def test_v42_gate_is_frozen_above_v41_untouched_mean():
    gate = load("configs/v42_broad_render_smoke_gate_v1.json")
    thresholds = gate["thresholds"]
    assert thresholds["minimum_mean_development_ood_safe_success"] == 0.65
    assert thresholds["minimum_worst_development_ood_safe_success"] == 0.30
    assert thresholds["minimum_nominal_safe_success"] == 0.85
    assert thresholds["minimum_intervention_safe_success"] == 0.90


def test_aggregate_understands_v42_non_ppo_accounting():
    source = (ROOT / "scripts/aggregate_visual_recovery.py").read_text()
    assert '"broad_render_dense_repair_v19": "dense_finetune_transitions"' in source
