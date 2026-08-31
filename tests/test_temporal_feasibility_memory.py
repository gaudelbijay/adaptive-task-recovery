"""Behavioral tests for conservative temporal goal memory."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

if not hasattr(torch, "tensor"):
    pytest.skip("real PyTorch is unavailable", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atr.feasibility.temporal_memory import (
    GoalStatus, TemporalGoalMemory, TemporalMemoryConfig,
)


def memory(minimum_observations=3):
    return TemporalGoalMemory(
        1, 2, device="cpu",
        config=TemporalMemoryConfig(
            unavailable_threshold=2.0, completion_threshold=2.0,
            evidence_decay=0.5, minimum_observations=minimum_observations,
        ),
    )


def test_single_transient_cannot_authorize_skipping():
    state = memory(minimum_observations=3)
    state.update(torch.tensor([[0.99, 0.01]]), torch.tensor([[0.01, 0.01]]))
    state.update(torch.tensor([[0.01, 0.01]]), torch.tensor([[0.01, 0.01]]))
    result = state.update(torch.tensor([[0.01, 0.01]]), torch.tensor([[0.01, 0.01]]))
    assert result.tolist() == [[GoalStatus.PENDING, GoalStatus.PENDING]]


def test_persistent_unavailability_becomes_monotone():
    state = memory()
    for _ in range(3):
        result = state.update(
            torch.tensor([[0.95, 0.01]]), torch.tensor([[0.01, 0.01]])
        )
    assert result.tolist() == [[GoalStatus.UNAVAILABLE, GoalStatus.PENDING]]
    for _ in range(5):
        result = state.update(
            torch.tensor([[0.01, 0.01]]), torch.tensor([[0.99, 0.01]])
        )
    assert result[0, 0] == GoalStatus.UNAVAILABLE


def test_completion_is_monotone_and_wins_same_frame_tie():
    state = memory(minimum_observations=1)
    result = state.update(torch.tensor([[0.99, 0.01]]), torch.tensor([[0.99, 0.01]]))
    assert result[0, 0] == GoalStatus.COMPLETED
    result = state.update(torch.tensor([[0.99, 0.01]]), torch.tensor([[0.01, 0.01]]))
    assert result[0, 0] == GoalStatus.COMPLETED


def test_masked_reset_affects_only_selected_batch_rows():
    state = TemporalGoalMemory(2, 2, device="cpu")
    state.status[:] = GoalStatus.UNAVAILABLE
    state.reset(torch.tensor([True, False]))
    assert state.status.tolist() == [
        [GoalStatus.PENDING, GoalStatus.PENDING],
        [GoalStatus.UNAVAILABLE, GoalStatus.UNAVAILABLE],
    ]
