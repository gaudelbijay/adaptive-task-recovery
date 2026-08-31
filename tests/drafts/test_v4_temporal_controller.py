"""Static leakage and protocol checks for the V4 temporal controller."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = (ROOT / "scripts/train_v4_temporal_feasibility_model.py").read_text()
EVAL = (ROOT / "scripts/evaluate_v4_temporal_controller.py").read_text()


def test_reverse_ejection_is_absent_from_classifier_training():
    assert 'TRAINING_KINDS = ("nominal", "ejection", "permanent_block", "temporary_block")' in TRAIN
    assert '"heldout_mechanism": "reverse_ejection"' in TRAIN


def test_controller_routes_without_oracle_action_input():
    assert 'late_horizon = int(metadata["horizon"])' in EVAL
    assert "if step == late_horizon" in EVAL
    assert "physical_family" in EVAL
    assert 'classifiers["onset"]' in EVAL
    assert "effective = torch.maximum(native, unavailable.float())" in EVAL
    assert "critic_goal_resolved" in EVAL  # scoring only
    route_section = EVAL.split("def routed_action", 1)[1].split("def main", 1)[0]
    assert "critic" not in route_section


def test_persistent_blockage_routes_to_state_specialist():
    assert "permanent-state-checkpoint" in EVAL
    assert "physical_family == 3" in EVAL
    assert "temporary_returning" in EVAL
    assert "blocker_target" in EVAL
    assert "blocker_engaged_observed" in EVAL
    assert "temporary_cleared_observed" in EVAL
    assert "BLOCKAGE_DECISION_HORIZON" in EVAL
    assert "retreat_action" in EVAL


def test_ejection_direction_uses_physical_sweeper_motion_not_label():
    assert '"critic_red_reverse_sweeper_pose"' in EVAL
    assert "physical_family == 2" in EVAL
    assert "MOTION_THRESHOLD" in EVAL
