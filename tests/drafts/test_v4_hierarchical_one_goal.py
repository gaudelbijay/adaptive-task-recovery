"""Static hierarchy checks."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/evaluate_v4_hierarchical_one_goal.py").read_text()


def test_remaining_goal_is_selected_only_from_task_progress():
    assert "active = torch.where(first_complete, 1 - original_first, original_first)" in SOURCE
    section = SOURCE.split("def hierarchical_observation", 1)[1].split("def main", 1)[0]
    assert "intervention_mechanism" not in section


def test_environment_task_memory_is_not_modified():
    assert 'extra["goal_progress"] = torch.zeros_like' in SOURCE
    assert "_completed" not in SOURCE
