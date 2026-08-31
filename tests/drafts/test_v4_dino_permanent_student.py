"""Static deployability checks for the spatial-DINO student pilot."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/train_v4_dino_permanent_student.py").read_text()


def test_actor_uses_rgb_proprio_and_explicit_progress_only():
    forward = SOURCE.split("def forward", 1)[1].split("def make_env", 1)[0]
    assert "critic" not in forward
    assert "spatial, cls, proprio, progress" in forward


def test_reverse_mechanism_or_label_is_not_an_actor_input():
    assert "critic_intervention_mechanism" not in SOURCE
