import json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"scripts"))
from v44_multiview_feature_agent import (
    AlwaysMultiViewFeatureV41Agent,
    CalibratedHybridFeatureV41Agent,
    MultiViewFeatureV41Agent,
)
load=lambda p:json.loads((ROOT/p).read_text())


def test_v44_has_frozen_v41_and_separate_renderer_encoder():
    agent=MultiViewFeatureV41Agent(64,23,10,7,True,0,0,True)
    assert agent.renderer_encoder is not agent.v41.base.encoder
    assert agent.route_threshold==0.9
    assert torch.sigmoid(agent.router[-1].bias).item()<0.02


def test_v44_budget_gate_and_unopened_suite():
    task=load("configs/visual_recovery_v44_multiview_feature_smoke.json")["experiments"][0]
    assert task["total_timesteps"]==task["feature_updates"]*task["num_envs"]*(1+len(task["paired_environment_profiles"]))
    gate=load("configs/v44_multiview_feature_smoke_gate_v1.json")["thresholds"]
    assert gate==load("configs/v43_identity_bounded_smoke_gate_v1.json")["thresholds"]
    assert load("configs/v42_confirmatory_unseen_visual_ood_v1.json")["seed_base"]==127000000


def test_v45_always_uses_renderer_encoder_and_preserves_gate():
    agent = AlwaysMultiViewFeatureV41Agent(64, 23, 10, 7, True, 0, 0, True)
    rgb = torch.zeros((3, 64, 64, 3), dtype=torch.uint8)
    expected = agent.renderer_latent(rgb)
    actual = agent.encode(rgb)
    assert torch.equal(actual, expected)
    assert agent.learned_route_fraction == 1.0

    task = load("configs/visual_recovery_v45_always_feature_smoke.json")["experiments"][0]
    assert task["deployment_mode"] == "always_feature"
    assert task["clean_feature_weight"] > 0
    assert task["clean_action_weight"] > 0
    gate = load("configs/v45_always_feature_smoke_gate_v1.json")["thresholds"]
    assert gate == load("configs/v44_multiview_feature_smoke_gate_v1.json")["thresholds"]


def test_v46_hybrid_threshold_accounting_and_gate_are_frozen():
    task = load("configs/visual_recovery_v46_hybrid_t050_smoke.json")["experiments"][0]
    assert task["route_threshold"] == 0.5
    assert task["total_timesteps"] == 512_000 + 307_200
    CalibratedHybridFeatureV41Agent.deployment_route_threshold = task["route_threshold"]
    agent = CalibratedHybridFeatureV41Agent(64, 23, 10, 7, True, 0, 0, True)
    assert agent.route_threshold == 0.5
    gate = load("configs/v46_hybrid_t050_smoke_gate_v1.json")["thresholds"]
    assert gate == load("configs/v45_always_feature_smoke_gate_v1.json")["thresholds"]
