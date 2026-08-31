import hashlib
import json
import math
from pathlib import Path

import torch

from v37_dense_canonical_agent import DenseCanonicalV19Agent


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / name).read_text())


def test_v37_budget_and_training_domains_are_explicit():
    config = load("configs/visual_recovery_v19_dense_canonical_v37_smoke.json")
    task = config["experiments"][0]
    assert task["total_timesteps"] == task["canonical_updates"] * task["num_envs"] * 5
    assert set(task["paired_environment_profiles"]) == {
        "camera_back_3cm", "camera_roll_right_2deg", "lighting_cool", "lighting_back_key",
    }


def test_v37_confirmation_domains_remain_disjoint_from_training():
    task = load("configs/visual_recovery_v19_dense_canonical_v37_smoke.json")["experiments"][0]
    confirm = load("configs/v36_confirmatory_unseen_visual_ood_v1.json")
    names = {item["name"] for item in confirm["variants"] if item["name"] not in {"baseline", "progress_cyclic_shift"}}
    assert names.isdisjoint(task["paired_environment_profiles"])


def test_dense_agent_is_always_on_and_zero_initialized():
    agent = DenseCanonicalV19Agent(64, 10, 10, 4, True, 0, 2, True)
    assert torch.count_nonzero(agent.dense_residual[-2].weight) == 0
    assert torch.count_nonzero(agent.dense_residual[-2].bias) == 0
    assert not hasattr(agent, "route_threshold")


def test_v37_gate_is_frozen_to_one_seed_and_strict_breadth():
    gate = load("configs/v37_dense_canonical_smoke_gate_v1.json")
    assert gate["matched_training_seed"] == 1788
    assert gate["thresholds"]["minimum_mean_development_ood_safe_success"] == 0.55
    assert gate["thresholds"]["minimum_worst_development_ood_safe_success"] == 0.30


def test_untouched_confirmation_spec_hash():
    path = ROOT / "configs/v36_confirmatory_unseen_visual_ood_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "9a3008b4323e7e0bf15e4e0b95fec913638896d88c6af061f4f97610d0b352fc"
