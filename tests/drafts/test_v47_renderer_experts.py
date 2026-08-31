import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from v47_renderer_expert_agent import RendererExpertV41Agent


def load(path):
    return json.loads((ROOT / path).read_text())


def test_v47_has_disjoint_experts_and_rgb_router():
    agent = RendererExpertV41Agent(64, 23, 10, 7, True, 0, 0, True)
    assert agent.camera_encoder is not agent.v41.base.encoder
    assert agent.lighting_encoder is not agent.v41.base.encoder
    assert agent.camera_encoder is not agent.lighting_encoder
    assert agent.router[-1].out_features == 3
    rgb = torch.zeros((2, 64, 64, 3), dtype=torch.uint8)
    assert agent.router_logits(rgb).shape == (2, 3)


def test_v47_budget_and_gate_are_unchanged():
    task = load("configs/visual_recovery_v47_renderer_expert_smoke.json")["experiments"][0]
    assert task["total_timesteps"] == task["expert_updates"] * task["num_envs"] * (
        1 + len(task["paired_environment_profiles"])
    )
    assert set(task["lighting_profiles"]) == {
        "lighting_bright_side", "lighting_green_ambient"
    }
    gate = load("configs/v47_renderer_expert_smoke_gate_v1.json")["thresholds"]
    assert gate == load("configs/v46_hybrid_t050_smoke_gate_v1.json")["thresholds"]
    assert load("configs/v42_confirmatory_unseen_visual_ood_v1.json")["seed_base"] == 127_000_000
