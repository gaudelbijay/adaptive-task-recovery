import json
from pathlib import Path

from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent
from v37_dense_canonical_agent import DenseCanonicalV19Agent


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v39_budget_threshold_and_backkey_oversampling():
    task = load("configs/visual_recovery_v19_backkey_v39_smoke.json")["experiments"][0]
    assert task["total_timesteps"] == task["fine_tune_updates"] * task["num_envs"] * 5
    assert task["magnitude_threshold"] == 0.003
    assert task["profile_sampling_cycle"].count("lighting_back_key") == 4
    assert len(task["profile_sampling_cycle"]) == 7


def test_magnitude_gate_has_no_new_checkpoint_parameters():
    source = DenseCanonicalV19Agent(64, 10, 10, 4, True, 0, 2, True)
    gated = MagnitudeGatedDenseV19Agent(64, 10, 10, 4, True, 0, 2, True)
    assert source.state_dict().keys() == gated.state_dict().keys()
    gated.load_state_dict(source.state_dict(), strict=True)
    assert gated.magnitude_threshold == 0.003


def test_v39_gate_is_unchanged_and_confirmation_untouched():
    gate = load("configs/v39_backkey_smoke_gate_v1.json")
    confirm = load("configs/v36_confirmatory_unseen_visual_ood_v1.json")
    assert gate["thresholds"]["minimum_nominal_safe_success"] == 0.85
    assert gate["thresholds"]["minimum_worst_development_ood_safe_success"] == 0.30
    assert confirm["seed_base"] == 117000000
