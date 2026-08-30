import ast
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v34_factorized_canonical_agent import FactorizedCanonicalV19Agent
from v35_translation_repair_agent import (
    TranslationRepairedV34Agent, sample_warp, synthesize_content_translation,
)


def test_v35_known_translation_round_trip():
    rgb = torch.randint(0, 256, (3, 64, 64, 3), dtype=torch.uint8)
    offsets = torch.tensor([[4.0, 0.0], [-3.0, 2.0], [0.0, -2.0]])
    shifted = synthesize_content_translation(rgb, offsets)
    restored = sample_warp(shifted, offsets)
    assert restored.shape == rgb.shape
    # Border replication makes only a narrow boundary irrecoverable.
    assert float((restored[:, 6:-6, 6:-6] - rgb[:, 6:-6, 6:-6]).abs().max()) < 2.5e-3


def test_v35_negative_route_is_exact_v34():
    torch.manual_seed(23)
    source = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    candidate = TranslationRepairedV34Agent(64, 19, 25, 7, True, 0, 14, True)
    candidate.initialize_from_v34(source.state_dict())
    with torch.no_grad():
        candidate.translation.shift_logit.weight.zero_()
        candidate.translation.shift_logit.bias.fill_(-100)
    rgb = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    expected = source.encode(rgb)
    actual = candidate.encode(rgb)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert candidate.learned_translation_route_fraction == 0.0


def test_v35_checkpoint_round_trip_is_strict():
    source = FactorizedCanonicalV19Agent(64, 19, 25, 7, True, 0, 14, True)
    first = TranslationRepairedV34Agent(64, 19, 25, 7, True, 0, 14, True)
    first.initialize_from_v34(source.state_dict())
    second = TranslationRepairedV34Agent(64, 19, 25, 7, True, 0, 14, True)
    second.load_state_dict(first.state_dict(), strict=True)


def test_v35_has_no_evaluator_domain_label_and_frozen_gate():
    evaluator = (SCRIPTS / "evaluate_v35_visual_recovery.py").read_text()
    tree = ast.parse(evaluator)
    assert not any(
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"environment_profile", "visual_perturbation"}
        for node in ast.walk(tree)
    )
    config = json.loads((
        ROOT / "configs/visual_recovery_v34_translation_repair_v35_smoke.json"
    ).read_text())
    task = config["experiments"][0]
    assert task["total_timesteps"] == task["translation_updates"] * task["num_envs"]
    assert [4, 0] in task["translation_training_offsets"]
    gate = json.loads((ROOT / "configs/v35_translation_repair_smoke_gate_v1.json").read_text())
    assert gate["thresholds"]["minimum_worst_ood_safe_success"] == 0.25
