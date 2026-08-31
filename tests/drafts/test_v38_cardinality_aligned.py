import json
from pathlib import Path

from train_v38_cardinality_aligned_canonical import cumulative_source_interactions


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v38_uses_distinct_single_camera_reference_and_exact_budget():
    task = load("configs/visual_recovery_v19_cardinality_aligned_v38_smoke.json")["experiments"][0]
    assert task["training_env_id"] == "LearnedRecovery-v3-MultiCamera"
    assert task["reference_env_id"] == "LearnedRecovery-v3"
    assert task["total_timesteps"] == task["canonical_updates"] * task["num_envs"] * 6
    assert task["synchronization_check_frequency"] == 20


def test_v38_source_accounting_prefers_cumulative_total():
    assert cumulative_source_interactions({
        "simulator_transitions": 256000,
        "total_simulator_transitions": 100255744,
    }) == 100255744


def test_v38_gate_and_confirmation_remain_separate():
    gate = load("configs/v38_cardinality_aligned_smoke_gate_v1.json")
    confirm = load("configs/v36_confirmatory_unseen_visual_ood_v1.json")
    assert gate["thresholds"]["minimum_worst_development_ood_safe_success"] == 0.30
    assert confirm["seed_base"] == 117000000
    assert "camera_right_back_4cm" in {item["name"] for item in confirm["variants"]}
