import sys
import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path("scripts").resolve()))
from train_v19_robust_distillation import robust_augment  # noqa: E402
from check_v27_robust_distill_smoke_gate import per_seed_rate  # noqa: E402


SMOKE = Path("configs/visual_recovery_v19_robust_distill_v27_smoke.json")
FULL = Path("configs/visual_recovery_v19_robust_distill_v27.json")
DEVELOPMENT = Path("configs/v27_smoke_development_ood_v1.json")
INCUMBENT = Path("configs/v19_incumbent_causal_ood_v1.json")
GATE = Path("configs/v27_robust_distill_smoke_gate_v1.json")


def test_robust_augmentation_is_deterministic_shape_preserving_and_nontrivial():
    rgb = torch.arange(4 * 16 * 16 * 3, dtype=torch.int64)
    rgb = (rgb % 256).to(torch.uint8).reshape(4, 16, 16, 3)
    kwargs = {
        "pad": 4, "brightness": (0.65, 1.05),
        "channel_gain": (0.8, 1.2), "probability": 1.0,
    }
    torch.manual_seed(19)
    first = robust_augment(rgb, **kwargs)
    torch.manual_seed(19)
    second = robust_augment(rgb, **kwargs)
    assert first.shape == (4, 3, 16, 16)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert not torch.equal(first, rgb.permute(0, 3, 1, 2).float().div(255))
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0


def test_zero_probability_returns_exact_normalized_pixels_and_validates_ranges():
    rgb = torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8)
    actual = robust_augment(
        rgb, pad=4, brightness=(0.5, 1.5), channel_gain=(0.5, 1.5),
        probability=0.0,
    )
    assert torch.equal(actual, rgb.permute(0, 3, 1, 2).float().div(255))
    with pytest.raises(ValueError, match="probability"):
        robust_augment(
            rgb, pad=0, brightness=(1, 1), channel_gain=(1, 1), probability=1.1,
        )


def test_smoke_and_full_configs_differ_only_in_identity_seed_and_budget():
    smoke = json.loads(SMOKE.read_text())
    full = json.loads(FULL.read_text())
    assert smoke["seeds"] == [1788]
    assert full["seeds"] == [9351, 4796, 1788]
    assert smoke["experiments"][0]["distillation_updates"] == 2000
    assert full["experiments"][0]["distillation_updates"] == 7500
    left, right = copy.deepcopy(smoke), copy.deepcopy(full)
    for payload in (left, right):
        payload.pop("name")
        payload.pop("seeds")
        payload.pop("claim_boundary")
        payload["experiments"][0].pop("method")
        payload["experiments"][0].pop("distillation_updates")
    assert left == right


def test_development_suite_and_gate_are_frozen_before_smoke_metrics():
    development = json.loads(DEVELOPMENT.read_text())
    incumbent = json.loads(INCUMBENT.read_text())
    for field in (
        "conditions", "episodes", "num_envs", "seed_base",
        "hypothesis_thresholds", "variants",
    ):
        assert development[field] == incumbent[field]
    gate = json.loads(GATE.read_text())
    assert gate["thresholds"] == {
        "minimum_nominal_baseline_safe_success": 0.85,
        "minimum_intervention_baseline_safe_success": 0.85,
        "minimum_mean_ood_safe_success_improvement": 0.20,
        "minimum_worst_ood_safe_success": 0.25,
        "maximum_individual_ood_regression": 0.05,
        "minimum_causal_safe_success_drop": 0.03,
        "require_positive_causal_lower_bound": True,
    }
    assert "not held-out" in gate["claim_boundary"]


def test_v27_gate_uses_the_matched_smoke_seed_not_the_pooled_rate():
    record = {
        "variant_safe_success_rate": 0.5,
        "per_seed": [
            {"training_seed": 9351, "variant_safe_success_rate": 0.1},
            {"training_seed": 4796, "variant_safe_success_rate": 0.2},
            {"training_seed": 1788, "variant_safe_success_rate": 0.9},
        ],
    }
    assert per_seed_rate(record, 1788) == 0.9
    with pytest.raises(ValueError, match="matched seed"):
        per_seed_rate(record, 71064)
