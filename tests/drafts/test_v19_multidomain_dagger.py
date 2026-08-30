import ast
import copy
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_visual_recovery_ppo import apply_visual_perturbation  # noqa: E402
from train_v19_multidomain_dagger import apply_sensor_shift  # noqa: E402


TRAINER = Path("scripts/train_v19_multidomain_dagger.py")
SMOKE = Path("configs/visual_recovery_v19_multidomain_dagger_v30_smoke.json")
FULL = Path("configs/visual_recovery_v19_multidomain_dagger_v30.json")
DEVELOPMENT = Path("configs/v30_smoke_development_ood_v1.json")
GATE = Path("configs/v30_multidomain_dagger_smoke_gate_v1.json")


def test_v30_uses_full_episode_domain_envs_and_routed_state_teachers():
    source = TRAINER.read_text()
    assert "paired_state_error" not in source
    assert "paired_segment_steps" not in source
    assert "dual_teacher_strict_route" in source
    assert "nominal_teacher.get_action" in source
    assert "strict_teacher.get_action" in source
    assert 'source_action if domain["profile"] == "nominal"' in source
    assert "ignore_terminations=True" not in source
    tree = ast.parse(source)
    step_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "step"
        and node.args and ast.unparse(node.args[0]) == "executed"
    ]
    assert len(step_calls) == 1
    assert ast.unparse(step_calls[0].args[0]) == "executed"


def test_v30_sensor_shifts_match_the_observed_evaluator():
    rgb = torch.arange(2 * 16 * 16 * 3, dtype=torch.int64)
    rgb = (rgb % 256).to(torch.uint8).reshape(2, 16, 16, 3)
    for mode in ("pixel_shift_right_4", "brightness_70", "warm_color_shift"):
        assert torch.equal(apply_sensor_shift(rgb, mode), apply_visual_perturbation(rgb, mode))


def test_v30_smoke_and_full_match_except_allocation_fields():
    smoke, full = json.loads(SMOKE.read_text()), json.loads(FULL.read_text())
    assert smoke["seeds"] == [1788]
    assert full["seeds"] == [9351, 4796, 1788]
    left, right = copy.deepcopy(smoke), copy.deepcopy(full)
    for payload in (left, right):
        payload.pop("name")
        payload.pop("seeds")
        payload.pop("claim_boundary")
        payload["experiments"][0].pop("method")
        payload["experiments"][0].pop("dagger_updates")
        payload["experiments"][0].pop("total_timesteps")
    assert left == right
    for config in (smoke, full):
        task = config["experiments"][0]
        assert task["total_timesteps"] == (
            task["dagger_updates"] * task["num_envs_per_domain"]
            * len(task["domain_profiles"])
        )


def test_v30_gate_reuses_frozen_thresholds_and_observed_suite():
    development, gate = json.loads(DEVELOPMENT.read_text()), json.loads(GATE.read_text())
    assert len(development["variants"]) == 11
    assert development["seed_base"] == 81000000
    assert gate["thresholds"] == {
        "minimum_nominal_baseline_safe_success": 0.85,
        "minimum_intervention_baseline_safe_success": 0.85,
        "minimum_mean_ood_safe_success_improvement": 0.20,
        "minimum_worst_ood_safe_success": 0.25,
        "maximum_individual_ood_regression": 0.05,
        "minimum_causal_safe_success_drop": 0.03,
        "require_positive_causal_lower_bound": True,
    }
    assert "Post-hoc" in development["claim_boundary"]
