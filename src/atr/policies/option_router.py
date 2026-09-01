"""Factorized belief models for recovery-option selection.

The routers deliberately contain no environment names, intervention IDs, or
geometry thresholds.  Each maps a prefix of matched physical observations to a
distribution over recovery options.  :class:`FactorizedOptionRouter` separates
event type, sweep direction, and blocker persistence into distinct heads so
each component can be audited; :class:`UnstructuredOptionGRU` is the
capacity-matched control with a single option head.

Every model here is *temporally causal*: at decision time it reads only
observations at or before the current step, never a future frame, and
:func:`current_centered_sequence` centers against the current frame rather than
the episode end.  That property is audited and holds.

These models do **not** infer causal dynamics, and the module was renamed away
from "causal" to stop implying they do.  Two results constrain the mechanism.
Reversing the prefix in time leaves held-out reverse accuracy at 97.7%, 77.6%,
and 96.9%, so temporal *direction* is not used.  And a single-observation model
reading one past frame (:class:`StaticOffsetRouter`) reaches 100% held-out
reverse accuracy, so mechanism identification does not require history at all.
What history buys is deciding whether an obstruction will clear: closed-loop,
both non-recurrent arms fail that confusion pair in opposite directions while
both recurrent arms solve it.  The supported claim is temporal aggregation of
signed motion evidence with deferred commitment.

**Wire-format identifiers are frozen.**  The `model` string stored inside a
checkpoint -- `"causal_gru"` for :class:`FactorizedOptionRouter` -- is not
renamed.  Frozen gate configs, `router_checkpoint_sha256` provenance for an
already-opened once-only confirmation, result manifests, and the gate audit
scripts all key on those exact strings.  Python identifiers describe the model;
the persisted string identifies a specific frozen artifact, and the two are
deliberately allowed to differ.
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


def deployable_option_targets(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deployable option targets without labels from future events."""
    condition = tensors["condition"]
    length = tensors["length"]
    onset = tensors["onset"]
    safe_option = torch.full_like(tensors["option"], 5)
    pre_event = length < onset + 2
    if "option_ready" in tensors:
        # Some task families expose a causal physical readiness predicate at
        # collection time (for example, blocker contact or completed return).
        # It is a training target only, never a router input. This avoids a
        # task-specific fixed wait while preserving the original V4 behavior
        # for datasets that do not contain the field.
        ready = tensors["option_ready"].bool()
        safe_option[pre_event] = 0
        safe_option[condition == 0] = 0
        safe_option[ready] = tensors["option"][ready]
        original_block = tensors["block_status"]
        block_status = torch.where(
            original_block == 1, torch.zeros_like(original_block),
            torch.where(
                original_block == 2, torch.ones_like(original_block),
                torch.full_like(original_block, -100),
            ),
        )
        return safe_option, block_status
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


class FactorizedOptionRouter(nn.Module):
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


class StaticOffsetRouter(nn.Module):
    """Single-observation baseline that reads one *past* frame, not the final one.

    :class:`StaticOptionRouter` reads the final frame, which current-centering
    forces to exactly zero -- so it receives no information by construction and
    cannot distinguish "history is required" from "this arm was handed zeros".
    This variant reads one earlier frame instead. Under current-centering that
    frame is the signed displacement between then and now, so the model gets a
    real motion signal from a single observation and no sequence model.

    ``offset=None`` reads the earliest valid frame, the maximum-information
    single frame. An integer ``offset`` reads that many steps back from the
    current frame, clamped to the start of the prefix.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 96, offset: int | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.offset = None if offset is None else int(offset)
        if self.offset is not None and self.offset < 1:
            raise ValueError("offset must be at least one frame in the past")
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

    def _select(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(sequence.device)
        if self.offset is None:
            index = torch.zeros_like(lengths)
        else:
            index = (lengths - 1 - self.offset).clamp_min(0)
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), index]

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> RouterOutput:
        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],), sequence.shape[1], dtype=torch.long,
                device=sequence.device,
            )
        frame = self._select(sequence, lengths)
        return self.heads(self.encoder((frame - self.input_mean) / self.input_scale))


class MomentSummaryRouter(nn.Module):
    """Non-recurrent control that summarises the whole prefix.

    The endpoint-pair and single-frame controls are weak: they read one or two
    observations. This one reads every frame but has no sequence encoder and no
    access to their order -- it takes the mean and standard deviation over the
    valid prefix. It is therefore the strongest *order-free* control, and it is
    the one that decides whether a ladder verdict is robust: a benchmark whose
    held-out mechanism falls to an order-free summary is not testing temporal
    composition, however far a single frame gets.

    Included so every benchmark is scored against an identical rung set. It
    mirrors the moment baseline already used on REBOOT.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.input_dim = int(input_dim)
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_scale", torch.ones(self.input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(2 * self.input_dim, hidden_dim), nn.SiLU(),
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
        normalized = (sequence - self.input_mean) / self.input_scale
        time = (
            torch.arange(sequence.shape[1], device=sequence.device)[None]
            < lengths.to(sequence.device)[:, None]
        ).float()[:, :, None]
        count = time.sum(1).clamp_min(1.0)
        mean = (normalized * time).sum(1) / count
        variance = (((normalized - mean[:, None]) ** 2) * time).sum(1) / count
        return self.heads(self.encoder(torch.cat((mean, variance.sqrt()), dim=1)))


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
