import json
from pathlib import Path

import torch

from v36_continuous_canonical_agent import (
    ContinuousCanonicalizer, invert_theta, similarity_theta, synthesize_corruption,
)


SMOKE = Path("configs/visual_recovery_v19_continuous_canonical_v36_smoke.json")
DEVELOPMENT = Path("configs/v36_smoke_development_ood_v1.json")
CONFIRMATION = Path("configs/v36_confirmatory_unseen_visual_ood_v1.json")
GATE = Path("configs/v36_continuous_canonical_smoke_gate_v1.json")


def test_v36_preserves_v19_and_has_exact_smoke_budget():
    config = json.loads(SMOKE.read_text())
    task = config["experiments"][0]
    assert task["source_visual_checkpoint"].endswith("dual_specialist_dagger_visual_ppo/seed_{seed}/best.pt")
    assert task["canonical_updates"] * task["num_envs"] == task["total_timesteps"] == 256000
    assert task["route_threshold"] == 0.9
    assert task["max_rotation_degrees"] > 2
    assert task["minimum_scale"] < 0.95 < task["maximum_scale"]


def test_v36_confirmation_is_frozen_and_disjoint_from_development():
    development = json.loads(DEVELOPMENT.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    development_names = {item["name"] for item in development["variants"]}
    confirmation_names = {item["name"] for item in confirmation["variants"]}
    assert development["seed_base"] != confirmation["seed_base"]
    assert development_names & confirmation_names == {"baseline", "progress_cyclic_shift"}
    assert len(confirmation_names) == 9
    assert confirmation["hypothesis_thresholds"]["minimum_ood_safe_success"] == 0.70


def test_position_aware_estimator_has_no_global_pool_and_clean_route_is_exact():
    module = ContinuousCanonicalizer()
    assert not any(isinstance(item, torch.nn.AdaptiveAvgPool2d) for item in module.modules())
    rgb = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
    corrected, logits, parameters, gain, bias = module.correct(rgb, hard_route=True)
    assert torch.equal(corrected, rgb.float())
    assert bool((torch.sigmoid(logits) < module.route_threshold).all())
    assert torch.equal(parameters, torch.zeros_like(parameters))
    assert torch.equal(gain, torch.ones_like(gain))
    assert torch.equal(bias, torch.zeros_like(bias))


def test_similarity_inverse_is_algebraically_exact():
    parameters = torch.tensor([[2.25, -1.5, 0.07, -0.03], [-3.0, 2.0, -0.1, 0.05]])
    theta = similarity_theta(parameters, 64, 64)
    inverse = invert_theta(theta)
    bottom = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 3).expand(2, -1, -1)
    product = torch.cat((theta, bottom), dim=1) @ torch.cat((inverse, bottom), dim=1)
    assert torch.allclose(product, torch.eye(3).expand(2, -1, -1), atol=1e-5)


def test_identity_synthesis_is_pixel_exact():
    rgb = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    parameters = torch.zeros((2, 4))
    gain = torch.ones((2, 3))
    bias = torch.zeros((2, 3))
    corrupted = synthesize_corruption(rgb, parameters, gain, bias)
    assert torch.allclose(corrupted, rgb.float(), atol=1e-3)


def test_smoke_gate_is_frozen_before_training():
    gate = json.loads(GATE.read_text())
    assert gate["matched_training_seed"] == 1788
    assert gate["thresholds"] == {
        "minimum_nominal_safe_success": 0.85,
        "minimum_intervention_safe_success": 0.85,
        "minimum_causal_safe_success_drop": 0.03,
        "require_positive_causal_lower_bound": True,
        "minimum_mean_development_ood_safe_success": 0.55,
        "minimum_worst_development_ood_safe_success": 0.30,
    }
