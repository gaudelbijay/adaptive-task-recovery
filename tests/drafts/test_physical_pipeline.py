"""Fast contract tests for the complete non-teleport pipeline."""

import numpy as np

from atr.physical_pipeline import RECOVERY_CHANGE_CROP, instruction_graph, recovery_change_score


def test_instruction_is_parsed_with_goals_and_constraints():
    graph = instruction_graph()
    assert [goal.target_object for goal in graph.goals] == ["potted_meat_can", "cracker_box"]
    assert {constraint.kind for constraint in graph.constraints} == {"never_move"}
    assert graph.constraints[0].tolerance == 0.05


def test_recovery_change_score_uses_only_declared_rgb_crop():
    before = np.zeros((512, 512, 3), dtype=np.uint8)
    after = before.copy()
    y0, y1, x0, x1 = RECOVERY_CHANGE_CROP
    after[y0:y1, x0:x1] = 10
    assert recovery_change_score(before, after) == 10.0

    outside = before.copy()
    outside[400:500, 400:500] = 255
    assert recovery_change_score(before, outside) == 0.0


def test_pipeline_module_does_not_import_teleport_executor():
    import atr.physical_pipeline as pipeline

    assert "attempt_goal" not in pipeline.__dict__
    assert pipeline.attempt_goal_with_real_grasp.__module__.endswith(
        "tidy_up_replicacad_manipulation"
    )
