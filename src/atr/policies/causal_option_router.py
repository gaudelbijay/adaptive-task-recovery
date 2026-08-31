"""Causal, factorized belief model for recovery-option selection.

The router deliberately contains no environment names, intervention IDs, or
geometry thresholds.  It maps a prefix of matched physical observations to a
distribution over recovery options.  Its factorization separates event type,
sweep direction, and blocker persistence so each component can be audited.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


OPTION_NAMES = (
    "nominal", "forward", "reverse", "permanent", "temporary_recovery", "defer",
)
EVENT_NAMES = ("none", "sweep", "block")
BLOCK_STATUS_NAMES = ("permanent", "cleared")


def causal_safe_targets(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deployable option targets without labels from future events."""
    condition = tensors["condition"]
    length = tensors["length"]
    onset = tensors["onset"]
    safe_option = torch.full_like(tensors["option"], 5)
    pre_event = length < onset + 2
    sweep_mature = length >= onset + 2
    safe_option[pre_event] = 0
    safe_option[condition == 0] = 0
    safe_option[(condition == 1) & sweep_mature] = 1
    safe_option[(condition == 4) & sweep_mature] = 2
    safe_option[(condition == 2) & (length >= onset + 52)] = 3
    safe_option[(condition == 3) & tensors["temporary_cleared"].bool()] = 4
    original_block = tensors["block_status"]
    block_status = torch.where(
        original_block == 1, torch.zeros_like(original_block),
        torch.where(
            original_block == 2, torch.ones_like(original_block),
            torch.full_like(original_block, -100),
        ),
    )
    return safe_option, block_status


@dataclass(frozen=True)
class RouterOutput:
    option_log_probability: torch.Tensor
    event_logits: torch.Tensor
    direction_logits: torch.Tensor
    block_status_logits: torch.Tensor
    readiness_logits: torch.Tensor

    @property
    def option_probability(self) -> torch.Tensor:
        return self.option_log_probability.exp()

    @property
    def option(self) -> torch.Tensor:
        return self.option_log_probability.argmax(dim=-1)


def _last_valid(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError("sequence must have shape [batch, time, features]")
    if lengths.shape != (sequence.shape[0],):
        raise ValueError("lengths must have shape [batch]")
    if bool(((lengths < 1) | (lengths > sequence.shape[1])).any()):
        raise ValueError("every length must be in [1, time]")
    index = lengths.to(sequence.device) - 1
    return sequence[torch.arange(sequence.shape[0], device=sequence.device), index]


def current_centered_sequence(
    sequence: torch.Tensor, lengths: torch.Tensor, geometry_dim: int,
) -> torch.Tensor:
    """Express every causal geometry frame relative to the current frame.

    The transform is causal at deployment because ``lengths - 1`` is the
    current observation, never a future frame. The current geometry becomes
    exactly zero, preventing a static model from recovering absolute pose.
    """
    if geometry_dim == 0:
        return sequence
    if geometry_dim < 0 or geometry_dim > sequence.shape[-1]:
        raise ValueError("geometry_dim must be within the feature width")
    current = _last_valid(sequence[..., :geometry_dim], lengths)
    centered = sequence.clone()
    centered[..., :geometry_dim] -= current[:, None, :]
    return centered


class _FactorizedHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.event = nn.Linear(hidden_dim, len(EVENT_NAMES))
        self.direction = nn.Linear(hidden_dim, 2)
        self.block_status = nn.Linear(hidden_dim, len(BLOCK_STATUS_NAMES))
        self.readiness = nn.Linear(hidden_dim, 2)

    def forward(self, latent: torch.Tensor) -> RouterOutput:
        event_logits = self.event(latent)
        direction_logits = self.direction(latent)
        block_status_logits = self.block_status(latent)
        readiness_logits = self.readiness(latent)
        event = event_logits.log_softmax(dim=-1)
        direction = direction_logits.log_softmax(dim=-1)
        block = block_status_logits.log_softmax(dim=-1)
        readiness = readiness_logits.log_softmax(dim=-1)

        # The mutually exclusive paths below form a normalized distribution:
        # defer; decide*{none; sweep*(forward|reverse);
        # block*(permanent|cleared)}.  The explicit readiness variable lets the
        # model abstain for *any* ambiguous event, rather than misusing one of
        # the physical event classes as a generic wait state.
        nominal = readiness[:, 1] + event[:, 0]
        temporary_recovery = readiness[:, 1] + event[:, 2] + block[:, 1]
        option_log_probability = torch.stack(
            (
                nominal,
                readiness[:, 1] + event[:, 1] + direction[:, 0],
                readiness[:, 1] + event[:, 1] + direction[:, 1],
                readiness[:, 1] + event[:, 2] + block[:, 0],
                temporary_recovery,
                readiness[:, 0],
            ),
            dim=-1,
        )
        return RouterOutput(
            option_log_probability=option_log_probability,
            event_logits=event_logits,
            direction_logits=direction_logits,
            block_status_logits=block_status_logits,
            readiness_logits=readiness_logits,
        )


class CausalOptionRouter(nn.Module):
    """GRU belief state with an auditable physical-event factorization."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, layers: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_scale", torch.ones(self.input_dim))
        self.encoder = nn.GRU(
            self.input_dim, self.hidden_dim, num_layers=self.layers,
            batch_first=True, dropout=0.1 if self.layers > 1 else 0.0,
        )
        self.heads = _FactorizedHeads(self.hidden_dim)

    def set_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != (self.input_dim,) or scale.shape != (self.input_dim,):
            raise ValueError("normalization vectors do not match input_dim")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp_min(1e-6))

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> RouterOutput:
        if sequence.ndim != 3 or sequence.shape[-1] != self.input_dim:
            raise ValueError("invalid sequence shape")
        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
                device=sequence.device,
            )
        normalized = (sequence - self.input_mean) / self.input_scale
        encoded, _ = self.encoder(normalized)
        return self.heads(_last_valid(encoded, lengths))


class StaticOptionRouter(nn.Module):
    """Matched-input baseline that sees only the final observation."""

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.input_dim = int(input_dim)
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_scale", torch.ones(self.input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.heads = _FactorizedHeads(hidden_dim)

    def set_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != (self.input_dim,) or scale.shape != (self.input_dim,):
            raise ValueError("normalization vectors do not match input_dim")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp_min(1e-6))

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> RouterOutput:
        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
                device=sequence.device,
            )
        final = _last_valid(sequence, lengths)
        return self.heads(self.encoder((final - self.input_mean) / self.input_scale))


class UnstructuredOptionGRU(nn.Module):
    """Capacity-matched temporal baseline without the causal factorization."""

    def __init__(self, input_dim: int, hidden_dim: int = 96, layers: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_scale", torch.ones(self.input_dim))
        self.encoder = nn.GRU(
            self.input_dim, hidden_dim, num_layers=layers, batch_first=True,
            dropout=0.1 if layers > 1 else 0.0,
        )
        self.option_head = nn.Linear(hidden_dim, len(OPTION_NAMES))

    def set_normalization(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if mean.shape != (self.input_dim,) or scale.shape != (self.input_dim,):
            raise ValueError("normalization vectors do not match input_dim")
        self.input_mean.copy_(mean)
        self.input_scale.copy_(scale.clamp_min(1e-6))

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
                device=sequence.device,
            )
        encoded, _ = self.encoder((sequence - self.input_mean) / self.input_scale)
        return self.option_head(_last_valid(encoded, lengths)).log_softmax(dim=-1)
