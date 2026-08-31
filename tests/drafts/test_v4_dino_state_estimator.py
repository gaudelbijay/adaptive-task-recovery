"""Static contract tests for visual state estimation."""

from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[2] / "scripts/train_v4_dino_state_estimator.py").read_text()


def test_deployment_composition_uses_predicted_physics_and_known_proprioception():
    compose = SOURCE.split("def compose_teacher_observation", 1)[1].split("class Spatial", 1)[0]
    assert '"agent": obs["agent"]' in compose
    assert '"goal_progress": extra["critic_goal_resolved"]' in compose
    assert "critic_intervention_mechanism" not in SOURCE


def test_estimator_not_action_head_is_supervised():
    assert "smooth_l1_loss(predicted_scaled, exact * scale)" in SOURCE
