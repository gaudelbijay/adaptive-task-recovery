import json
from pathlib import Path

from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent
from v41_magnitude_gated_agent import V41MagnitudeGatedDenseV19Agent


ROOT = Path(__file__).resolve().parents[2]


def load(path): return json.loads((ROOT / path).read_text())


def test_v41_changes_no_checkpoint_parameters_and_freezes_threshold():
    old = MagnitudeGatedDenseV19Agent(64, 10, 10, 4, True, 0, 2, True)
    new = V41MagnitudeGatedDenseV19Agent(64, 10, 10, 4, True, 0, 2, True)
    assert old.state_dict().keys() == new.state_dict().keys()
    new.load_state_dict(old.state_dict(), strict=True)
    assert new.magnitude_threshold == 0.015


def test_v41_uses_audited_v40_config_and_unchanged_gate():
    spec = load("configs/v41_smoke_development_ood_v1.json")
    gate = load("configs/v41_threshold015_smoke_gate_v1.json")
    old_gate = load("configs/v40_backkey_smoke_gate_v1.json")
    assert spec["policy_configs"]["threshold015_v40"].endswith("v40_smoke.json")
    assert gate["thresholds"] == old_gate["thresholds"]
    assert gate["matched_training_seed"] == 1788
