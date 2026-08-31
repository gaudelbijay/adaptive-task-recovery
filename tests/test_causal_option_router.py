import torch

from atr.policies.causal_option_router import (
    CausalOptionRouter, StaticOptionRouter, UnstructuredOptionGRU,
    causal_safe_targets,
)


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


def test_normalization_rejects_wrong_feature_contract():
    model = CausalOptionRouter(4, hidden_dim=8, layers=1)
    try:
        model.set_normalization(torch.zeros(5), torch.ones(5))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong normalization shape was accepted")
