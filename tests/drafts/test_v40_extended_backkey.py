import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v40_changes_only_identity_budget_cycle_and_boundary():
    v39 = load("configs/visual_recovery_v19_backkey_v39_smoke.json")
    v40 = load("configs/visual_recovery_v19_backkey_v40_smoke.json")
    assert v40["learning_rate"] == v39["learning_rate"]
    assert v40["weight_decay"] == v39["weight_decay"]
    left = dict(v39["experiments"][0]); right = dict(v40["experiments"][0])
    for key in ("method", "fine_tune_updates", "total_timesteps", "profile_sampling_cycle"):
        left.pop(key); right.pop(key)
    assert left == right
    task = v40["experiments"][0]
    assert task["total_timesteps"] == task["fine_tune_updates"] * task["num_envs"] * 5
    assert task["profile_sampling_cycle"].count("lighting_back_key") == 8
    assert len(task["profile_sampling_cycle"]) == 11


def test_v40_gate_remains_identical_to_v39_thresholds():
    v39 = load("configs/v39_backkey_smoke_gate_v1.json")
    v40 = load("configs/v40_backkey_smoke_gate_v1.json")
    assert v40["thresholds"] == v39["thresholds"]
    assert v40["matched_training_seed"] == v39["matched_training_seed"] == 1788
