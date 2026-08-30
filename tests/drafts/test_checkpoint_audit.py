"""Fail-closed contracts for post-training checkpoint integrity audits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
AUDITOR = ROOT / "scripts/audit_training_checkpoints.py"


def _module():
    spec = importlib.util.spec_from_file_location("audit_training_checkpoints", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_nested_finite_tensor_audit_counts_model_and_optimizer_state():
    count, failures = _module()._finite_tensors({
        "agent": {"weight": torch.ones(2, 3)},
        "optimizer": {"state": [{"moment": torch.zeros(3)}]},
        "integer_counter": torch.tensor([4]),
    })
    assert count == 2
    assert failures == []


def test_nonfinite_tensor_audit_reports_exact_nested_paths():
    count, failures = _module()._finite_tensors({
        "agent": {"weight": torch.tensor([1.0, float("nan")])},
        "optimizer": {"state": [{"moment": torch.tensor([float("inf")])}]},
    })
    assert count == 2
    assert failures == ["agent.weight", "optimizer.state[0].moment"]


def test_auditor_requires_exact_floor_aligned_budget_and_source_provenance():
    source = AUDITOR.read_text(encoding="utf-8")
    assert 'latest_step != scheduled' in source
    assert 'best_step <= latest_step' in source
    assert 'source.get("trainer")' in source
    assert 'source.get("environment")' in source
    assert 'optimizer_count == 0 or optimizer_failures' in source
