import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_shift_action_stability_smoke_gate as gate  # noqa: E402
from check_drac_stability_smoke_gate import normalized  # noqa: E402


def load(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_v24_smoke_and_extension_are_matched_v19_treatments():
    baseline = load("visual_recovery_dual_specialist_dagger_v19.json")
    smoke = load("visual_recovery_dual_specialist_shift_action_v24_smoke.json")
    full = load("visual_recovery_dual_specialist_shift_action_v24.json")
    assert normalized(baseline, remove_budget=True) == normalized(
        smoke, remove_budget=True
    )
    assert normalized(baseline, remove_budget=False) == normalized(
        full, remove_budget=False
    )
    for task in (smoke["experiments"][0], full["experiments"][0]):
        assert task["augmentation_pad"] == 4
        assert task["drac_policy_coefficient"] == 0.1


def test_gate_records_every_distinct_method_source():
    assert set(gate.SMOKE_SOURCE_PATHS) == {
        "trainer_wrapper", "base_trainer", "environment", "environment_v3",
        "bounded_shift_action_consistency",
    }
    assert set(gate.EXTENSION_SOURCE_PATHS) == {
        "trainer", "trainer_wrapper", "base_trainer", "environment",
        "environment_v3", "bounded_shift_action_consistency",
    }
    for path in (*gate.SMOKE_SOURCE_PATHS.values(), *gate.EXTENSION_SOURCE_PATHS.values()):
        assert path.is_file()
    smoke_hash = __import__("hashlib").sha256(
        gate.SMOKE_SOURCE_PATHS["trainer_wrapper"].read_bytes()
    ).hexdigest()
    assert smoke_hash == "037c5403aac7a911f32e5bc4b185aa2ebe65604ae2d407ad349e22b6ca23cd39"
    extension_hash = __import__("hashlib").sha256(
        gate.EXTENSION_SOURCE_PATHS["trainer_wrapper"].read_bytes()
    ).hexdigest()
    assert extension_hash == "77cd6312c6ab9f0515647618963f8b5d2a7f160ea2cbb10fcf419551d6e9434b"


def test_frozen_gate_keeps_best_tail_and_safety_checks():
    config = load("shift_action_stability_smoke_gate_v1.json")
    assert config["tail_evaluations"] == 3
    assert config["thresholds"] == {
        "minimum_best_success_at_end": 0.9,
        "maximum_best_constraint_violation": 0.05,
        "minimum_best_score_margin": -0.05,
        "maximum_tail_mean_constraint_violation": 0.05,
        "minimum_tail_mean_score_improvement": 0.05,
    }
    assert "not a performance or robustness claim" in config["claim_boundary"]


def test_bounded_loss_check_requires_finite_in_range_records(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join([
        json.dumps({"global_step": 10, "train_loss": {"drac_policy": 0.2}}),
        json.dumps({"global_step": 20, "train_loss": {"drac_policy": 1.95}}),
        json.dumps({"global_step": 30, "train_loss": {"drac_policy": 9.0}}),
    ]) + "\n")
    result = gate.bounded_loss_check(path, 20)
    assert result == {
        "finite_bounded_consistency": True,
        "maximum_logged_consistency_loss": 1.95,
        "consistency_records": 2,
    }
    assert gate.bounded_loss_check(path, 30)["finite_bounded_consistency"] is False
