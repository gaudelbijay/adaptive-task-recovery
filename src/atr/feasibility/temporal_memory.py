"""Monotone temporal memory for goal completion and infeasibility.

Per-frame visual classifiers are noisy and cannot distinguish an irreversible
failure from a transient obstruction at event onset.  This module turns
calibrated per-goal probabilities into a conservative task state using a
sequential log-likelihood ratio.  It can defer, but once a goal is physically
completed or declared unavailable the corresponding state is monotone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class GoalStatus(IntEnum):
    PENDING = 0
    COMPLETED = 1
    UNAVAILABLE = 2


@dataclass(frozen=True)
class TemporalMemoryConfig:
    unavailable_threshold: float = 4.0
    completion_threshold: float = 4.0
    evidence_decay: float = 0.9
    minimum_observations: int = 3
    probability_clip: float = 1e-4

    def __post_init__(self):
        if self.unavailable_threshold <= 0 or self.completion_threshold <= 0:
            raise ValueError("decision thresholds must be positive")
        if not 0 <= self.evidence_decay <= 1:
            raise ValueError("evidence_decay must be in [0, 1]")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if not 0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must be in (0, 0.5)")


class TemporalGoalMemory:
    """Batchable three-state goal memory with conservative evidence fusion."""

    def __init__(
        self, batch_size: int, goal_count: int, *, device: torch.device | str,
        config: TemporalMemoryConfig = TemporalMemoryConfig(),
    ):
        if batch_size < 1 or goal_count < 1:
            raise ValueError("batch_size and goal_count must be positive")
        self.config = config
        shape = (batch_size, goal_count)
        self.status = torch.full(shape, GoalStatus.PENDING, dtype=torch.long, device=device)
        self.unavailable_evidence = torch.zeros(shape, device=device)
        self.completion_evidence = torch.zeros(shape, device=device)
        self.observations = torch.zeros(shape, dtype=torch.long, device=device)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        if mask is None:
            mask = torch.ones(self.status.shape[0], dtype=torch.bool, device=self.status.device)
        self.status[mask] = GoalStatus.PENDING
        self.unavailable_evidence[mask] = 0
        self.completion_evidence[mask] = 0
        self.observations[mask] = 0

    def _log_odds(self, probability: torch.Tensor) -> torch.Tensor:
        clip = self.config.probability_clip
        probability = probability.clamp(clip, 1 - clip)
        return torch.logit(probability)

    def update(
        self,
        unavailable_probability: torch.Tensor,
        completion_probability: torch.Tensor,
    ) -> torch.Tensor:
        if unavailable_probability.shape != self.status.shape:
            raise ValueError("unavailable_probability has the wrong shape")
        if completion_probability.shape != self.status.shape:
            raise ValueError("completion_probability has the wrong shape")
        pending = self.status == GoalStatus.PENDING
        decay = self.config.evidence_decay
        self.unavailable_evidence[pending] = (
            decay * self.unavailable_evidence[pending]
            + self._log_odds(unavailable_probability[pending])
        )
        self.completion_evidence[pending] = (
            decay * self.completion_evidence[pending]
            + self._log_odds(completion_probability[pending])
        )
        self.observations[pending] += 1
        eligible = pending & (self.observations >= self.config.minimum_observations)

        # Completion wins a same-frame tie: once the requested relation has
        # actually been achieved, a later visual disappearance must not
        # rewrite history as an infeasible goal.
        completed = eligible & (
            self.completion_evidence >= self.config.completion_threshold
        )
        unavailable = (
            eligible
            & ~completed
            & (self.unavailable_evidence >= self.config.unavailable_threshold)
        )
        self.status[completed] = GoalStatus.COMPLETED
        self.status[unavailable] = GoalStatus.UNAVAILABLE
        return self.status.clone()

    @property
    def resolved(self) -> torch.Tensor:
        return self.status != GoalStatus.PENDING

    @property
    def unavailable(self) -> torch.Tensor:
        return self.status == GoalStatus.UNAVAILABLE
