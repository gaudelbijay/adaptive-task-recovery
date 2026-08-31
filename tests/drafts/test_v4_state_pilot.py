"""Allocation and leakage checks for the V4 state-teacher pilot."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pilot_holds_reverse_ejection_out_and_has_one_seed():
    config = json.loads((ROOT / "configs/learned_recovery_v4_state_pilot.json").read_text())
    assert config["seeds"] == [9351]
    task = config["experiments"][0]
    assert task["total_timesteps"] == 25_000_000
    assert task["env_kwargs"]["intervention_types"] == [
        "ejection", "permanent_block", "temporary_block",
    ]
    assert "reverse_ejection" not in json.dumps(task["env_kwargs"])


def test_state_observation_has_blocker_pose_but_not_mechanism_label():
    source = (ROOT / "src/atr/envs/learned_recovery_v4.py").read_text()
    state_section = source.split('if "state" in self.obs_mode:', 1)[1]
    assert '"red_goal_blocker_pose"' in state_section
    assert '"blue_goal_blocker_pose"' in state_section
    assert '"intervention_mechanism"' not in state_section
