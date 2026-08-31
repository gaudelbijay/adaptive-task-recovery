import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from v48_geometry_expert_agent import HierarchicalGeometryExpertAgent


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v48_is_hierarchical_and_preserves_v47_experts():
    agent = HierarchicalGeometryExpertAgent(64, 23, 10, 7, True, 0, 0, True)
    assert agent.geometry_encoder is not agent.v47.v41.base.encoder
    assert agent.geometry_router[-1].out_features == 1
    assert agent.geometry_threshold == 0.5
    rgb = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
    assert agent.geometry_logits(rgb).shape == (2,)


def test_v48_budget_gate_and_unopened_suite():
    task = load("configs/visual_recovery_v48_geometry_expert_smoke.json")["experiments"][0]
    assert task["total_timesteps"] == task["geometry_updates"] * task["num_envs"]
    gate = load("configs/v48_geometry_expert_smoke_gate_v1.json")["thresholds"]
    assert gate == load("configs/v47_renderer_expert_smoke_gate_v1.json")["thresholds"]
    assert load("configs/v42_confirmatory_unseen_visual_ood_v1.json")["seed_base"] == 127_000_000
