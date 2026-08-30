import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path("scripts").resolve()))
from evaluate_visual_recovery_ppo import apply_visual_perturbation  # noqa: E402
from train_v19_multidomain_distillation import apply_sensor_shift  # noqa: E402


V28 = Path("scripts/train_v19_rendered_domain_distillation.py")
V29 = Path("scripts/train_v19_multidomain_distillation.py")
SMOKE = Path("configs/visual_recovery_v19_multidomain_distill_v29_smoke.json")
FULL = Path("configs/visual_recovery_v19_multidomain_distill_v29.json")
DEVELOPMENT = Path("configs/v29_smoke_development_ood_v1.json")
GATE = Path("configs/v29_multidomain_distill_smoke_gate_v1.json")
V28_UNSEEN = Path("configs/v28_unseen_visual_ood_v1.json")
V29_UNSEEN = Path("configs/v29_unseen_visual_ood_v1.json")
V28_FINAL_GATE = Path("configs/v28_final_release_gate_v1.json")
V29_FINAL_GATE = Path("configs/v29_final_release_gate_v1.json")


def test_v28_provenance_source_is_immutable_and_v29_is_distinct():
    assert hashlib.sha256(V28.read_bytes()).hexdigest() == (
        "3d67313aa8ee5f5cbde05b7245f3dfef933f35a71f1204e6195a5df3dcac512c"
    )
    assert V28.read_bytes() != V29.read_bytes()


def test_v29_freezes_policy_heads_and_anchors_teacher_features():
    source = V29.read_text()
    assert "for parameter in agent.parameters():" in source
    assert "for parameter in agent.encoder.parameters():" in source
    assert "feature_anchor_loss" in source
    assert "teacher_reference_latent" in source
    tree = ast.parse(source)
    optimizer_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.optim.Adam"
    ]
    assert len(optimizer_calls) == 1
    assert "parameter.requires_grad" in ast.unparse(optimizer_calls[0])


def test_v29_training_sensor_shifts_exactly_match_observed_evaluator():
    rgb = torch.arange(2 * 16 * 16 * 3, dtype=torch.int64)
    rgb = (rgb % 256).to(torch.uint8).reshape(2, 16, 16, 3)
    for mode in ("pixel_shift_right_4", "brightness_70", "warm_color_shift"):
        assert torch.equal(apply_sensor_shift(rgb, mode), apply_visual_perturbation(rgb, mode))


def test_v29_smoke_and_full_are_matched_except_allocation_fields():
    smoke, full = json.loads(SMOKE.read_text()), json.loads(FULL.read_text())
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
    task = smoke["experiments"][0]
    assert task["freeze_policy_heads"] is True
    assert task["ignore_terminations_during_pairing"] is True
    assert "ignore_terminations=True" in V29.read_text()
    assert set(task["sensor_augmentations"]) == {
        "pixel_shift_right_4", "brightness_70", "warm_color_shift",
    }


def test_v29_development_gate_is_frozen_and_reuses_only_observed_suite():
    development = json.loads(DEVELOPMENT.read_text())
    gate = json.loads(GATE.read_text())
    assert development["seed_base"] == 81000000
    assert len(development["variants"]) == 11
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


def test_v29_inherits_v28_unseen_suite_and_final_thresholds_exactly():
    old, new = json.loads(V28_UNSEEN.read_text()), json.loads(V29_UNSEEN.read_text())
    for payload in (old, new):
        payload.pop("name")
        payload.pop("selection")
        payload.pop("policy_configs")
        payload.pop("claim_boundary")
    assert old == new
    assert json.loads(V28_FINAL_GATE.read_text())["thresholds"] == (
        json.loads(V29_FINAL_GATE.read_text())["thresholds"]
    )
