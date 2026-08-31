"""Protocol checks for resolved-progress V4 DAgger."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_teacher_receives_resolved_not_only_completed_progress():
    source = (ROOT / "scripts/train_v4_resolved_progress_dagger.py").read_text()
    assert '"goal_progress": base.visual_progress_target(obs)' in source
    assert "base.reconstruct_state_teacher_observation =" in source


def test_pilot_holds_out_temporary_and_reverse_mechanisms():
    config = json.loads((ROOT / "configs/visual_recovery_v4_resolved_dagger_pilot.json").read_text())
    task = config["experiments"][0]
    assert config["seeds"] == [9351]
    assert task["env_kwargs"]["intervention_types"] == ["ejection", "permanent_block"]
    assert task["total_timesteps"] == 10_000_000
