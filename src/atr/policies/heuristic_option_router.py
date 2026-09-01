"""Hand-written motion-threshold router, the V28 baseline as a matched router.

Unlike :mod:`atr.policies.option_router`, this module intentionally
contains environment feature names and a geometry threshold.  That is the
point of the baseline: it is the hand-engineered state machine the learned
routers must beat, expressed over the *same* matched observation tensor so the
comparison is input-matched rather than mechanism-matched.

The router reads only the current-centered causal prefix.  It never sees a
mechanism ID, an intervention target, a future frame, or `critic_goal_resolved`.
Because every geometry frame is expressed relative to the current frame, the
largest absolute value inside a mechanism's feature slice is that mechanism's
observed motion extent over the prefix.
"""

from __future__ import annotations

import torch
from torch import nn

from .option_router import OPTION_NAMES

# Motion threshold in metres, taken unchanged from the frozen V28 controller
# (`scripts/evaluate_v4_temporal_controller.py`, MOTION_THRESHOLD).
MOTION_THRESHOLD = 0.005

# Feature-name prefixes that identify each mechanism actor's own pose.  Slices
# are resolved from the router metadata at construction so a schema change
# fails loudly instead of silently reading the wrong columns.
FORWARD_SWEEPER_PREFIX = "critic_red_sweeper_pose."
FORWARD_SWEEPER_PREFIX_ALT = "critic_blue_sweeper_pose."
REVERSE_SWEEPER_PREFIX = "critic_red_reverse_sweeper_pose."
REVERSE_SWEEPER_PREFIX_ALT = "critic_blue_reverse_sweeper_pose."
BLOCKER_PREFIX = "critic_red_goal_blocker_pose."
BLOCKER_PREFIX_ALT = "critic_blue_goal_blocker_pose."

OPTION_NOMINAL = OPTION_NAMES.index("nominal")
OPTION_FORWARD = OPTION_NAMES.index("forward")
OPTION_REVERSE = OPTION_NAMES.index("reverse")
OPTION_PERMANENT = OPTION_NAMES.index("permanent")
OPTION_TEMPORARY = OPTION_NAMES.index("temporary_recovery")


def resolve_indices(feature_names: list[str], *prefixes: str) -> list[int]:
    """Return the column indices whose names start with any given prefix."""
    index = [
        i for i, name in enumerate(feature_names)
        if any(name.startswith(p) for p in prefixes)
    ]
    if not index:
        raise ValueError(f"no features matched prefixes {prefixes}")
    return index


class HeuristicMotionRouter(nn.Module):
    """Motion-threshold state machine over the matched centered prefix.

    The decision order mirrors the frozen V28 controller: a blocker event
    dominates a sweep event, and a reverse sweep is distinguished from a
    forward sweep by which sweeper actor moved.  Permanence is decided by
    observed physical evidence only -- a blocker that moved earlier in the
    prefix but is close to its current pose again is treated as a temporary
    obstruction, matching V28's "wait for causal evidence" rule.
    """

    def __init__(self, feature_names: list[str], threshold: float = MOTION_THRESHOLD):
        super().__init__()
        self.threshold = float(threshold)
        self.register_buffer(
            "forward_index",
            torch.tensor(resolve_indices(
                feature_names, FORWARD_SWEEPER_PREFIX, FORWARD_SWEEPER_PREFIX_ALT,
            ), dtype=torch.long),
        )
        self.register_buffer(
            "reverse_index",
            torch.tensor(resolve_indices(
                feature_names, REVERSE_SWEEPER_PREFIX, REVERSE_SWEEPER_PREFIX_ALT,
            ), dtype=torch.long),
        )
        self.register_buffer(
            "blocker_index",
            torch.tensor(resolve_indices(
                feature_names, BLOCKER_PREFIX, BLOCKER_PREFIX_ALT,
            ), dtype=torch.long),
        )

    @staticmethod
    def _extent(sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Largest absolute centered displacement anywhere in the prefix."""
        return sequence[..., index].abs().amax(dim=(1, 2))

    @staticmethod
    def _initial_offset(sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """How far the actor's earliest observed pose sits from its current one.

        Under current-centering both a persistent and a returned blocker show
        near-zero motion in the most recent frames, so recency cannot separate
        them.  The starting pose can: an actor that moved in and stayed ends
        far from where it began, while one that moved and returned does not.
        """
        return sequence[:, 0, :][..., index].abs().amax(dim=1)

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        blocker = self._extent(sequence, self.blocker_index)
        reverse = self._extent(sequence, self.reverse_index)
        forward = self._extent(sequence, self.forward_index)
        blocker_start = self._initial_offset(sequence, self.blocker_index)

        option = torch.full(
            (sequence.shape[0],), OPTION_NOMINAL, dtype=torch.long,
            device=sequence.device,
        )
        # A sweep event: whichever sweeper actor moved further wins.
        swept = (reverse > self.threshold) | (forward > self.threshold)
        option = torch.where(
            swept & (reverse >= forward),
            torch.full_like(option, OPTION_REVERSE), option,
        )
        option = torch.where(
            swept & (reverse < forward),
            torch.full_like(option, OPTION_FORWARD), option,
        )
        # A blocker event dominates a sweep. Permanence is decided by observed
        # physical evidence only: a blocker whose starting pose differs from
        # its current pose moved in and stayed; one that is back where it
        # began has cleared.
        blocked = blocker > self.threshold
        displaced = blocker_start > self.threshold
        option = torch.where(
            blocked & displaced, torch.full_like(option, OPTION_PERMANENT), option,
        )
        option = torch.where(
            blocked & ~displaced, torch.full_like(option, OPTION_TEMPORARY), option,
        )
        # Emit a hard one-hot log-probability so the evaluator's shared
        # confidence/abstention path treats this exactly like a learned router.
        logits = torch.full(
            (sequence.shape[0], len(OPTION_NAMES)), -30.0, device=sequence.device,
        )
        logits.scatter_(1, option[:, None], 0.0)
        return logits.log_softmax(dim=-1)


# ---------------------------------------------------------------------------
# Generic signed-axis variant, used for benchmarks whose ejection direction is
# a sign along one axis rather than which of two actors moved.
# ---------------------------------------------------------------------------

PEG_EJECTION_PREFIX = "hole_frame.peg_to_hole."
PEG_BLOCKER_PREFIX = "hole_frame.blocker_to_hole."
PEG_LATERAL_AXIS = "hole_frame.peg_to_hole.y"


class SignedAxisMotionRouter(nn.Module):
    """Motion-threshold router whose direction comes from the sign of one axis.

    `LearnedRecovery-v4` distinguishes forward from reverse ejection by *which*
    sweeper actor moved. `PegInsertionSide-v1` instead reflects the ejection
    impulse along the hole frame's lateral axis, so direction is the sign of a
    single coordinate. This router is the same hand-written rung of the control
    ladder expressed for that geometry: it reads the identical matched tensor,
    no mechanism ID and no future frame.

    Under current-centering, `centered[0] = raw[start] - raw[now]`. A peg pushed
    toward +y therefore has a *negative* centered value at the prefix start, so
    the sign is read and inverted rather than taken directly.
    """

    def __init__(
        self,
        feature_names: list[str],
        ejection_prefix: str = PEG_EJECTION_PREFIX,
        blocker_prefix: str = PEG_BLOCKER_PREFIX,
        lateral_feature: str = PEG_LATERAL_AXIS,
        threshold: float = MOTION_THRESHOLD,
    ):
        super().__init__()
        self.threshold = float(threshold)
        if lateral_feature not in feature_names:
            raise ValueError(f"missing lateral feature {lateral_feature!r}")
        self.register_buffer(
            "ejection_index",
            torch.tensor(resolve_indices(feature_names, ejection_prefix), dtype=torch.long),
        )
        self.register_buffer(
            "blocker_index",
            torch.tensor(resolve_indices(feature_names, blocker_prefix), dtype=torch.long),
        )
        self.register_buffer(
            "lateral_index",
            torch.tensor(feature_names.index(lateral_feature), dtype=torch.long),
        )

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape [batch, time, features]")
        ejection = sequence[..., self.ejection_index].abs().amax(dim=(1, 2))
        blocker = sequence[..., self.blocker_index].abs().amax(dim=(1, 2))
        blocker_start = sequence[:, 0, :][..., self.blocker_index].abs().amax(dim=1)
        # Negative centered start means the peg travelled toward +lateral.
        lateral_start = sequence[:, 0, self.lateral_index]

        option = torch.full(
            (sequence.shape[0],), OPTION_NOMINAL, dtype=torch.long,
            device=sequence.device,
        )
        ejected = ejection > self.threshold
        option = torch.where(
            ejected & (lateral_start < 0),
            torch.full_like(option, OPTION_FORWARD), option,
        )
        option = torch.where(
            ejected & (lateral_start >= 0),
            torch.full_like(option, OPTION_REVERSE), option,
        )
        blocked = blocker > self.threshold
        displaced = blocker_start > self.threshold
        option = torch.where(
            blocked & displaced, torch.full_like(option, OPTION_PERMANENT), option,
        )
        option = torch.where(
            blocked & ~displaced, torch.full_like(option, OPTION_TEMPORARY), option,
        )
        logits = torch.full(
            (sequence.shape[0], len(OPTION_NAMES)), -30.0, device=sequence.device,
        )
        logits.scatter_(1, option[:, None], 0.0)
        return logits.log_softmax(dim=-1)
