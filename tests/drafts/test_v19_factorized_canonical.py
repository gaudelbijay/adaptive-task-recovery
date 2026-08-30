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
from v34_factorized_canonical_agent import FactorizedCanonicalV19Agent


def test_v34_base_route_is_bit_exact_v19():
    torch.manual_seed(19)
    base = VisualAgent(64, 19, 25, 7, True, 0, 14, True)
    candidate = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    candidate.initialize_from_v19(base.state_dict())
    with torch.no_grad():
        final = candidate.router.network[-1]
        final.weight.zero_()
        final.bias.fill_(-100)
        final.bias[0] = 100
    rgb = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.randn(2, 19)
    expected = base.get_action(rgb, proprio, deterministic=True)
    latent = candidate.encode(rgb)
    progress = torch.sigmoid(candidate.goal_progress_predictor(latent))
    actual = torch.tanh(candidate.actor(torch.cat((latent, proprio, progress), dim=1)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate.learned_route_fraction == 0.0


def test_v34_factorized_canonicalizer_is_bounded_and_differentiable():
    candidate = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    rgb = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    canonical, flow = candidate.canonicalize(rgb, return_flow=True)
    assert canonical.shape == rgb.shape
    assert flow.shape == (2, 2, 64, 64)
    assert float(canonical.detach().min()) >= 0.0
    assert float(canonical.detach().max()) <= 255.0
    assert float(flow.detach().abs().max()) <= 8.0
    (canonical.mean() + flow.square().mean()).backward()
    assert candidate.canonicalizer.flow_head.weight.grad is not None
    assert candidate.canonicalizer.residual_head.weight.grad is not None


def test_v34_checkpoint_round_trip_is_strict():
    base = VisualAgent(64, 19, 25, 7, True, 0, 14, True)
    first = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    first.initialize_from_v19(base.state_dict())
    second = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    second.load_state_dict(first.state_dict(), strict=True)


def test_v34_uses_all_observed_domains_and_no_evaluator_label():
    trainer = (SCRIPTS / "train_v19_factorized_canonical.py").read_text()
    evaluator = (SCRIPTS / "evaluate_v34_visual_recovery.py").read_text()
    assert "geometry_rgbs.extend(apply_sensor_shift(base_rgb, mode) for mode in augmentations)" in trainer
    assert "lighting_observations[profile]" in trainer
    assert "enumerate(domain_router_logits, start=1)" in trainer
    tree = ast.parse(evaluator)
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"environment_profile", "visual_perturbation"}
        for node in ast.walk(tree)
    )


def test_v34_smoke_and_gate_contracts():
    config = json.loads((
        ROOT / "configs/visual_recovery_v19_factorized_canonical_v34_smoke.json"
    ).read_text())
    task = config["experiments"][0]
    assert config["seeds"] == [1788]
    assert task["total_timesteps"] == task["dagger_updates"] * task["num_envs"]
    assert task["paired_lighting_profiles"] == ["lighting_dim", "lighting_warm"]
    assert task["action_weight"] > 0 and task["flow_smoothness_weight"] > 0
    gate = json.loads((
        ROOT / "configs/v34_factorized_canonical_smoke_gate_v1.json"
    ).read_text())
    assert gate["thresholds"]["minimum_worst_ood_safe_success"] == 0.25
    assert gate["thresholds"]["minimum_mean_ood_safe_success_improvement"] == 0.20
