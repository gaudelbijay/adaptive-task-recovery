import pytest
import torch

from atr.policies.causal_option_router import OPTION_NAMES, current_centered_sequence
from atr.policies.heuristic_option_router import (
    MOTION_THRESHOLD, HeuristicMotionRouter, resolve_indices,
)


FEATURES = (
    [f"critic_red_sweeper_pose.goal_relative.{a}" for a in "xyz"]
    + [f"critic_blue_sweeper_pose.goal_relative.{a}" for a in "xyz"]
    + [f"critic_red_reverse_sweeper_pose.goal_relative.{a}" for a in "xyz"]
    + [f"critic_blue_reverse_sweeper_pose.goal_relative.{a}" for a in "xyz"]
    + [f"critic_red_goal_blocker_pose.goal_relative.{a}" for a in "xyz"]
    + [f"critic_blue_goal_blocker_pose.goal_relative.{a}" for a in "xyz"]
    + ["instruction.0", "instruction.1"]
)
GEOMETRY_DIM = 18
OPTION = {name: i for i, name in enumerate(OPTION_NAMES)}


def _prefix(moving_slice=None, magnitude=0.05, time=8, returns=False):
    """Build a raw prefix where one mechanism slice moves and then is centered."""
    sequence = torch.zeros(1, time, len(FEATURES))
    if moving_slice is not None:
        # The actor sits displaced early in the prefix and arrives at its
        # final pose by the last frame, so centering leaves a real signal.
        sequence[0, : time // 2, moving_slice] = magnitude
        if returns:
            sequence[0, time // 2 :, moving_slice] = 0.0
    lengths = torch.tensor([time])
    return current_centered_sequence(sequence, lengths, GEOMETRY_DIM), lengths


def _router():
    return HeuristicMotionRouter(FEATURES)


def test_feature_slices_resolve_to_the_named_mechanism_actors():
    router = _router()
    assert router.forward_index.tolist() == [0, 1, 2, 3, 4, 5]
    assert router.reverse_index.tolist() == [6, 7, 8, 9, 10, 11]
    assert router.blocker_index.tolist() == [12, 13, 14, 15, 16, 17]


def test_missing_features_fail_loudly_instead_of_reading_wrong_columns():
    with pytest.raises(ValueError):
        resolve_indices(["unrelated.feature"], "critic_red_sweeper_pose.")
    with pytest.raises(ValueError):
        HeuristicMotionRouter(["instruction.0", "instruction.1"])


def test_a_still_world_routes_to_nominal():
    router = _router()
    sequence, lengths = _prefix(moving_slice=None)
    assert router(sequence, lengths).argmax(-1).item() == OPTION["nominal"]


def test_forward_sweeper_motion_routes_to_the_forward_option():
    router = _router()
    sequence, lengths = _prefix(moving_slice=slice(0, 6))
    assert router(sequence, lengths).argmax(-1).item() == OPTION["forward"]


def test_reverse_sweeper_motion_routes_to_the_held_out_reverse_option():
    router = _router()
    sequence, lengths = _prefix(moving_slice=slice(6, 12))
    assert router(sequence, lengths).argmax(-1).item() == OPTION["reverse"]


def test_persistent_blocker_motion_routes_to_permanent():
    router = _router()
    sequence, lengths = _prefix(moving_slice=slice(12, 18))
    assert router(sequence, lengths).argmax(-1).item() == OPTION["permanent"]


def test_blocker_that_returns_routes_to_temporary_recovery():
    router = _router()
    # The blocker starts at rest, intrudes mid-prefix, then returns to where
    # it began. Its starting pose therefore matches its current pose, which is
    # what separates a cleared obstruction from a persistent one.
    sequence = torch.zeros(1, 8, len(FEATURES))
    sequence[0, 3:6, 12:18] = 0.05
    lengths = torch.tensor([8])
    centered = current_centered_sequence(sequence, lengths, GEOMETRY_DIM)
    assert centered[0, 0, 12:18].abs().max() == 0.0
    assert centered[0, :, 12:18].abs().max() > MOTION_THRESHOLD
    assert router(centered, lengths).argmax(-1).item() == OPTION["temporary_recovery"]


def test_permanent_and_temporary_blockers_are_separated_by_starting_pose():
    """Both look identical in recent frames; only the prefix start differs."""
    router = _router()
    lengths = torch.tensor([8])
    persistent = torch.zeros(1, 8, len(FEATURES))
    persistent[0, :4, 12:18] = 0.05          # moved in and stayed
    cleared = torch.zeros(1, 8, len(FEATURES))
    cleared[0, 3:6, 12:18] = 0.05            # moved in and returned
    for raw, expected in ((persistent, "permanent"), (cleared, "temporary_recovery")):
        centered = current_centered_sequence(raw, lengths, GEOMETRY_DIM)
        assert centered[0, -1, 12:18].abs().max() == 0.0  # recent motion identical
        assert router(centered, lengths).argmax(-1).item() == OPTION[expected]


def test_motion_below_the_frozen_threshold_stays_nominal():
    router = _router()
    sequence, lengths = _prefix(moving_slice=slice(0, 6), magnitude=MOTION_THRESHOLD / 2)
    assert router(sequence, lengths).argmax(-1).item() == OPTION["nominal"]


def test_output_is_a_normalized_log_probability_over_all_options():
    router = _router()
    sequence, lengths = _prefix(moving_slice=slice(6, 12))
    logp = router(sequence, lengths)
    assert logp.shape == (1, len(OPTION_NAMES))
    assert torch.allclose(logp.exp().sum(-1), torch.ones(1), atol=1e-5)
    assert logp.exp().max().item() > 0.99


def test_router_reads_no_future_frame():
    """Truncating the prefix must not change a decision already determined."""
    router = _router()
    full = torch.zeros(1, 12, len(FEATURES))
    full[0, :4, 6:12] = 0.05
    short_len, full_len = torch.tensor([8]), torch.tensor([12])
    short = current_centered_sequence(full[:, :8], short_len, GEOMETRY_DIM)
    assert router(short, short_len).argmax(-1).item() == OPTION["reverse"]


def test_batch_decisions_are_independent():
    router = _router()
    batch = torch.zeros(3, 8, len(FEATURES))
    batch[0, :4, 0:6] = 0.05
    batch[1, :4, 6:12] = 0.05
    lengths = torch.tensor([8, 8, 8])
    centered = current_centered_sequence(batch, lengths, GEOMETRY_DIM)
    assert router(centered, lengths).argmax(-1).tolist() == [
        OPTION["forward"], OPTION["reverse"], OPTION["nominal"],
    ]
