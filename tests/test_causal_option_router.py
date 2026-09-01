import pytest
import torch

from atr.policies.causal_option_router import (
    OPTION_NAMES, CausalOptionRouter, StaticOptionRouter, UnstructuredOptionGRU,
    causal_safe_targets, current_centered_sequence,
)
from atr.policies.peg_router_features import relative_geometry, world_to_local


def test_delayed_targets_never_encode_a_future_event():
    tensors = {
        "condition": torch.tensor([1, 1, 2, 2, 3]),
        "length": torch.tensor([12, 14, 63, 64, 64]),
        "onset": torch.tensor([12, 12, 12, 12, 12]),
        "option": torch.zeros(5, dtype=torch.long),
        "temporary_cleared": torch.tensor([False, False, False, False, True]),
        "block_status": torch.full((5,), -100, dtype=torch.long),
    }
    option, _ = causal_safe_targets(tensors)
    assert option.tolist() == [0, 1, 5, 3, 4]


def test_task_specific_readiness_is_training_only_and_causal():
    tensors = {
        "condition": torch.tensor([0, 1, 1, 2, 3]),
        "length": torch.tensor([20, 12, 16, 30, 70]),
        "onset": torch.tensor([12, 12, 12, 12, 12]),
        "option": torch.tensor([0, 1, 1, 3, 4]),
        "option_ready": torch.tensor([True, False, True, True, True]),
        "temporary_cleared": torch.tensor([False, False, False, False, True]),
        "block_status": torch.tensor([-100, -100, -100, 1, 2]),
    }
    option, block = causal_safe_targets(tensors)
    assert option.tolist() == [0, 0, 1, 3, 4]
    assert block.tolist() == [-100, -100, -100, 0, 1]


def test_factorized_router_is_normalized_and_finite():
    model = CausalOptionRouter(11, hidden_dim=16, layers=1).eval()
    output = model(torch.randn(7, 9, 11))
    assert torch.isfinite(output.option_log_probability).all()
    assert torch.allclose(
        output.option_probability.sum(dim=1), torch.ones(7), atol=1e-6,
    )


def test_router_is_causal_for_explicit_prefix_lengths():
    torch.manual_seed(4)
    model = CausalOptionRouter(6, hidden_dim=12, layers=1).eval()
    prefix = torch.randn(3, 5, 6)
    future_a = torch.cat((prefix, torch.randn(3, 4, 6)), dim=1)
    future_b = torch.cat((prefix, 100 * torch.randn(3, 4, 6)), dim=1)
    lengths = torch.full((3,), 5)
    a = model(future_a, lengths).option_log_probability
    b = model(future_b, lengths).option_log_probability
    assert torch.equal(a, b)


def test_matched_baselines_accept_identical_contract():
    sequence = torch.randn(5, 8, 13)
    lengths = torch.tensor([1, 2, 4, 7, 8])
    static = StaticOptionRouter(13, hidden_dim=16)(sequence, lengths)
    recurrent = UnstructuredOptionGRU(13, hidden_dim=16, layers=1)(sequence, lengths)
    assert static.option_log_probability.shape == recurrent.shape == (5, 6)


def test_external_static_baseline_is_parameter_matched():
    models = {
        "causal": CausalOptionRouter(13, hidden_dim=96, layers=2),
        "static": StaticOptionRouter(13, hidden_dim=288),
        "unstructured": UnstructuredOptionGRU(13, hidden_dim=96, layers=2),
    }
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    assert abs(counts["static"] - counts["causal"]) / counts["causal"] < 0.02
    assert abs(counts["unstructured"] - counts["causal"]) / counts["causal"] < 0.01


def test_normalization_rejects_wrong_feature_contract():
    model = CausalOptionRouter(4, hidden_dim=8, layers=1)
    try:
        model.set_normalization(torch.zeros(5), torch.ones(5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong normalization shape was accepted")


def test_current_centered_geometry_is_causal_and_zero_at_current_frame():
    sequence = torch.tensor([
        [[1.0, 3.0, 10.0], [4.0, 5.0, 11.0], [8.0, 9.0, 12.0]],
        [[2.0, 4.0, 20.0], [7.0, 8.0, 21.0], [99.0, 99.0, 99.0]],
    ])
    lengths = torch.tensor([3, 2])
    centered = current_centered_sequence(sequence, lengths, geometry_dim=2)
    assert torch.equal(centered[0, 2, :2], torch.zeros(2))
    assert torch.equal(centered[1, 1, :2], torch.zeros(2))
    assert torch.equal(centered[0, :, 2], sequence[0, :, 2])
    assert torch.equal(centered[1, :2, :2], torch.tensor([[-5.0, -4.0], [0.0, 0.0]]))


def test_current_centering_ignores_padded_future_when_length_is_shorter():
    prefix = torch.randn(2, 4, 5)
    a = torch.cat((prefix, torch.zeros(2, 3, 5)), dim=1)
    b = torch.cat((prefix, 100 * torch.randn(2, 3, 5)), dim=1)
    lengths = torch.full((2,), 4)
    assert torch.equal(
        current_centered_sequence(a, lengths, 3)[:, :4],
        current_centered_sequence(b, lengths, 3)[:, :4],
    )


def test_peg_router_vectors_are_rotated_into_randomized_hole_frame():
    root_half = 2 ** -0.5
    quaternion = torch.tensor([[root_half, 0.0, 0.0, root_half]])
    local = world_to_local(torch.tensor([[0.0, 1.0, 0.0]]), quaternion)
    assert torch.allclose(local, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)
    raw = torch.tensor([[
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, root_half, 0.0, 0.0, root_half,
        0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ]])
    geometry = relative_geometry(raw)
    assert torch.allclose(geometry[0, :3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_static_offset_router_reads_a_past_frame_not_the_zeroed_current_one():
    """The current frame is zero after centering; an earlier frame is not."""
    from atr.policies.causal_option_router import StaticOffsetRouter

    raw = torch.zeros(1, 8, 6)
    raw[0, :4, :] = 0.05
    lengths = torch.tensor([8])
    centered = current_centered_sequence(raw, lengths, 6)
    assert centered[0, -1].abs().max() == 0.0

    first = StaticOffsetRouter(6, 16, None)
    assert first._select(centered, lengths).abs().max() > 0.0

    zeroed = StaticOptionRouter(6, 16)
    from atr.policies.causal_option_router import _last_valid
    assert _last_valid(centered, lengths).abs().max() == 0.0


def test_static_offset_router_offset_is_clamped_to_the_prefix_start():
    from atr.policies.causal_option_router import StaticOffsetRouter

    sequence = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    lengths = torch.tensor([4])
    far = StaticOffsetRouter(3, 8, 99)
    assert torch.equal(far._select(sequence, lengths), sequence[:, 0])
    near = StaticOffsetRouter(3, 8, 1)
    assert torch.equal(near._select(sequence, lengths), sequence[:, 2])


def test_static_offset_router_rejects_a_non_past_offset():
    from atr.policies.causal_option_router import StaticOffsetRouter

    with pytest.raises(ValueError):
        StaticOffsetRouter(3, 8, 0)


def test_static_offset_router_emits_the_factorized_head_contract():
    from atr.policies.causal_option_router import StaticOffsetRouter

    model = StaticOffsetRouter(6, 16, None)
    output = model(torch.randn(4, 8, 6), torch.tensor([8, 8, 8, 8]))
    assert output.option_log_probability.shape == (4, len(OPTION_NAMES))
    assert output.event_logits.shape[0] == 4
