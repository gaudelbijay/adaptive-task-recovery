"""Static held-out and observation-adapter checks."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/evaluate_v3_state_expert_on_v4.py").read_text()


def test_adapter_removes_only_v4_blockers_not_physical_cube_state():
    assert '"red_goal_blocker_pose", "blue_goal_blocker_pose"' in SOURCE
    exclusion = SOURCE.split("if key not in", 1)[1].split("}", 1)[0]
    assert "red_cube_pose" not in exclusion


def test_reverse_ejection_is_explicitly_held_out():
    assert 'CONDITIONS = ("nominal", "ejection", "reverse_ejection")' in SOURCE
    assert '"heldout_mechanism": condition == "reverse_ejection"' in SOURCE


def test_canonicalization_uses_physical_state_not_mechanism_label():
    adapter = SOURCE.split("def v3_observation", 1)[1].split("def main", 1)[0]
    assert "pose[:, 0].abs() > 0.36" in adapter
    assert "intervention_mechanism" not in adapter
