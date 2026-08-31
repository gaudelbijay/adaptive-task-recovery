"""Static contract checks for V4 permanent-block teacher distillation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "scripts/train_v4_permanent_visual_dagger.py").read_text()


def test_v4_teacher_observation_includes_physical_blockers_not_mechanism_id():
    assert '"critic_red_goal_blocker_pose"' in SOURCE
    assert '"critic_blue_goal_blocker_pose"' in SOURCE
    assert "critic_intervention_mechanism" not in SOURCE
