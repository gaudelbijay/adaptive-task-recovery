import ast
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_visual_recovery_dual_teacher_ppo import VisualAgent
from v32_hybrid_domain_agent import HybridDomainAgent


def test_hybrid_base_route_is_exact_v19_computation():
    torch.manual_seed(7)
    base = VisualAgent(64, 19, 25, 7, True, 0, 14, True)
    hybrid = HybridDomainAgent(64, 19, 25, 7, True, 0, 14, True)
    hybrid.initialize_from_v19(base.state_dict())
    with torch.no_grad():
        hybrid.router[-1].weight.zero_()
        hybrid.router[-1].bias.fill_(-100)
    rgb = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.randn(3, 19)
    expected = base.get_action(rgb, proprio, deterministic=True)
    latent = hybrid.encode(rgb)
    progress = torch.sigmoid(hybrid.goal_progress_predictor(latent))
    actual = torch.tanh(hybrid.actor(torch.cat((latent, proprio, progress), dim=1)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert hybrid.learned_route_fraction == 0.0


def test_hybrid_checkpoint_round_trip_is_strict():
    base = VisualAgent(64, 19, 25, 7, True, 0, 14, True)
    first = HybridDomainAgent(64, 19, 25, 7, True, 0, 14, True)
    first.initialize_from_v19(base.state_dict())
    second = HybridDomainAgent(64, 19, 25, 7, True, 0, 14, True)
    second.load_state_dict(first.state_dict(), strict=True)


def test_v32_actor_never_receives_evaluation_domain_label():
    evaluator = (SCRIPTS / "evaluate_v32_visual_recovery.py").read_text()
    tree = ast.parse(evaluator)
    assert "HybridDomainAgent" in evaluator
    assert "domain_label_available_to_actor" in evaluator
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"environment_profile", "visual_perturbation"}
        for node in ast.walk(tree)
    )


def test_v32_geometry_loss_is_bounded_before_reduction():
    trainer = (SCRIPTS / "train_v19_geometry_router.py").read_text()
    assert "geometry_target[:, :12].div_(5.0)" in trainer
    assert "torch.tanh(agent.robust.privileged_predictor(latent))" in trainer
    assert "F.smooth_l1_loss" in trainer


def test_v32_smoke_contract_and_gate_are_frozen():
    config = json.loads((
        ROOT / "configs/visual_recovery_v19_geometry_router_v32_smoke.json"
    ).read_text())
    task = config["experiments"][0]
    assert config["seeds"] == [1788]
    assert task["total_timesteps"] == task["dagger_updates"] * task["num_envs"]
    assert task["camera_keys"] == [
        "base_camera", "camera_left_5cm", "camera_high_5cm",
    ]
    assert task["geometry_weight"] > 0
    assert task["router_weight"] > 0
    gate = json.loads((
        ROOT / "configs/v32_geometry_router_smoke_gate_v1.json"
    ).read_text())
    assert gate["thresholds"] == {
        "minimum_nominal_baseline_safe_success": 0.85,
        "minimum_intervention_baseline_safe_success": 0.85,
        "minimum_mean_ood_safe_success_improvement": 0.20,
        "minimum_worst_ood_safe_success": 0.25,
        "maximum_individual_ood_regression": 0.05,
        "minimum_causal_safe_success_drop": 0.03,
        "require_positive_causal_lower_bound": True,
    }
