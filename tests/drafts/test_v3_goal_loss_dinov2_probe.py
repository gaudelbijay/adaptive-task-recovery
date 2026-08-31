"""Lightweight contracts for the V3 goal-loss perception probe."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mani_skill")
pytest.importorskip("sklearn")


_PATH = Path(__file__).resolve().parents[2] / "scripts/probe_v3_goal_loss_dinov2.py"
_SPEC = importlib.util.spec_from_file_location("probe_v3_goal_loss_dinov2", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def test_goal_condition_builds_disjoint_interaction_blocks():
    features = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    conditioned = _MODULE.goal_condition(features)
    assert conditioned.shape == (2, 2, 6)
    assert torch.equal(conditioned[0, 0], torch.tensor([2.0, 3.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.equal(conditioned[0, 1], torch.tensor([0.0, 0.0, 2.0, 3.0, 0.0, 1.0]))


def test_goal_condition_allows_linear_goal_specific_decisions():
    # One visual scalar means opposite labels for the two queried goals.  This
    # XOR-like relation is impossible with [feature, goal_one_hot] additively,
    # but linearly separable after explicit interaction blocks.
    features = torch.tensor([[1.0], [-1.0]])
    rows = _MODULE.goal_condition(features).reshape(4, -1)
    weights = torch.tensor([1.0, -1.0, 0.0, 0.0])
    logits = rows @ weights
    assert torch.equal(logits > 0, torch.tensor([True, False, False, True]))
