import ast
import copy
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_visual_recovery_unseen_ood import (  # noqa: E402
    NEW_PERTURBATIONS,
    NEW_PROFILES,
    apply_visual_perturbation,
    camera_eye,
    lighting_parameters,
)


TRAINER = Path("scripts/train_v19_rendered_domain_distillation.py")
ADAPTER = Path("scripts/evaluate_v28_visual_recovery.py")
DEVELOPMENT_RUNNER = Path("scripts/run_v28_development_visual_ood.py")
SMOKE = Path("configs/visual_recovery_v19_render_distill_v28_smoke.json")
FULL = Path("configs/visual_recovery_v19_render_distill_v28.json")
DEVELOPMENT = Path("configs/v28_smoke_development_ood_v1.json")
UNSEEN = Path("configs/v28_unseen_visual_ood_v1.json")
GATE = Path("configs/v28_render_distill_smoke_gate_v1.json")
STRICT = Path("configs/strict_removal_v19_render_distill_v16.json")


def test_rendered_distillation_executes_identical_actions_and_checks_state_pairing():
    tree = ast.parse(TRAINER.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    step_calls = [
        node for node in calls
        if (
            isinstance(node.func, ast.Attribute) and node.func.attr == "step"
            and node.args and ast.unparse(node.args[0]) == "executed"
        )
    ]
    assert len(step_calls) == 2
    source = TRAINER.read_text()
    assert "paired_state_error" in source
    assert "paired state diverged" in source
    assert "paired_segment_steps" in source
    assert "reset_error > maximum_error" in source
    assert '"environment": hashlib.sha256' in source
    assert '"rendered_environment": hashlib.sha256' in source


def test_rendered_smoke_and_full_configs_are_matched_except_allocation_fields():
    smoke = json.loads(SMOKE.read_text())
    full = json.loads(FULL.read_text())
    assert smoke["seeds"] == [1788]
    assert full["seeds"] == [9351, 4796, 1788]
    left, right = copy.deepcopy(smoke), copy.deepcopy(full)
    for payload in (left, right):
        payload.pop("name")
        payload.pop("seeds")
        payload.pop("claim_boundary")
        payload["experiments"][0].pop("method")
        payload["experiments"][0].pop("distillation_updates")
        payload["experiments"][0].pop("total_timesteps")
    assert left == right
    experiment = smoke["experiments"][0]
    assert 0 < experiment["paired_segment_steps"] < experiment["env_kwargs"]["intervention_steps"]


def test_unseen_profiles_and_sensor_shifts_are_disjoint_and_nontrivial():
    development = json.loads(DEVELOPMENT.read_text())
    unseen = json.loads(UNSEEN.read_text())
    development_profiles = {
        item["environment_profile"] for item in development["variants"]
        if item["environment_profile"] != "nominal"
    }
    unseen_profiles = {
        item["environment_profile"] for item in unseen["variants"]
        if item["environment_profile"] != "nominal"
    }
    assert unseen_profiles == set(NEW_PROFILES)
    assert development_profiles.isdisjoint(unseen_profiles)
    assert set(NEW_PERTURBATIONS) == {
        "pixel_shift_left_3", "brightness_85", "cool_color_shift",
    }
    assert unseen["seed_base"] != development["seed_base"]
    rgb = torch.arange(2 * 8 * 8 * 3, dtype=torch.int64)
    rgb = (rgb % 256).to(torch.uint8).reshape(2, 8, 8, 3)
    for mode in NEW_PERTURBATIONS:
        changed = apply_visual_perturbation(rgb, mode)
        assert changed.shape == rgb.shape
        assert not torch.equal(changed, rgb)
    assert camera_eye("camera_right_3cm")[1] < 0
    assert camera_eye("camera_low_3cm")[2] < 0.72
    assert lighting_parameters("lighting_bright") != lighting_parameters("lighting_cool")


def test_v28_gate_and_unseen_thresholds_are_frozen_before_training():
    gate = json.loads(GATE.read_text())
    assert gate["thresholds"]["minimum_mean_ood_safe_success_improvement"] == 0.20
    assert gate["thresholds"]["minimum_worst_ood_safe_success"] == 0.25
    unseen = json.loads(UNSEEN.read_text())
    assert unseen["hypothesis_thresholds"]["minimum_ood_safe_success"] == 0.75
    assert unseen["hypothesis_thresholds"]["maximum_ood_safe_success_drop"] == 0.15
    assert "Pre-training unseen" in unseen["claim_boundary"]
    strict = json.loads(STRICT.read_text())
    assert [item["label"] for item in strict["cohorts"]] == [
        "v19_visual_incumbent", "v28_render_distilled_visual",
    ]
    assert strict["cohorts"][1]["config"] == str(FULL)


def test_v28_evaluation_adapter_removes_false_ppo_accounting():
    source = ADAPTER.read_text()
    assert 'payload["ppo_accounting_applicable"] = False' in source
    assert 'payload["distillation_student_transitions"]' in source
    assert 'payload["distillation_simulator_transitions"]' in source
    assert 'payload[key] = 0' in source
    development_source = DEVELOPMENT_RUNNER.read_text()
    assert 'evaluator_script="evaluate_v28_visual_recovery.py"' in development_source
    assert 'execution_protocol="V28 observed-suite development execution"' in development_source
