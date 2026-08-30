import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from drac_policy_consistency import (  # noqa: E402
    diagonal_gaussian_kl,
    drac_policy_consistency,
)


def test_identical_gaussians_have_zero_kl():
    mean = torch.tensor([[0.2, -0.3], [0.5, 0.1]])
    logstd = torch.tensor([[-0.4, 0.2], [-0.1, -0.7]])
    loss = diagonal_gaussian_kl(mean, logstd, mean.clone(), logstd.clone())
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)


def test_known_unit_variance_mean_shift_matches_exact_kl():
    target = torch.zeros((3, 2))
    augmented = torch.ones((3, 2), requires_grad=True)
    logstd = torch.zeros((3, 2), requires_grad=True)
    loss = diagonal_gaussian_kl(target, logstd, augmented, logstd)
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-7, rtol=0)


def test_target_is_stop_gradient_but_augmented_policy_receives_gradient():
    target_mean = torch.randn(4, 3, requires_grad=True)
    target_logstd = torch.randn(4, 3, requires_grad=True)
    augmented_mean = torch.randn(4, 3, requires_grad=True)
    augmented_logstd = torch.randn(4, 3, requires_grad=True)
    diagonal_gaussian_kl(
        target_mean, target_logstd, augmented_mean, augmented_logstd,
    ).backward()
    assert target_mean.grad is None
    assert target_logstd.grad is None
    assert augmented_mean.grad is not None
    assert augmented_logstd.grad is not None
    assert torch.isfinite(augmented_mean.grad).all()
    assert torch.isfinite(augmented_logstd.grad).all()


class TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.goal_progress_predictor = torch.nn.Linear(4, 2)
        self.actor = torch.nn.Linear(4 + 2 + 2, 2)
        self.actor_logstd = torch.nn.Parameter(torch.full((1, 2), -0.5))

    def encode(self, rgb, augment=False):
        value = rgb.float().mean(dim=(1, 2)) / 255
        if augment:
            value = torch.roll(value, shifts=1, dims=1)
        return self.encoder(value)


def test_agent_consistency_loss_is_finite_positive_and_differentiable():
    torch.manual_seed(9)
    agent = TinyAgent()
    rgb = torch.randint(0, 256, (8, 4, 4, 3), dtype=torch.uint8)
    proprio = torch.randn(8, 2)
    loss = drac_policy_consistency(agent, rgb, proprio)
    assert torch.isfinite(loss)
    assert loss > 0
    loss.backward()
    assert agent.encoder.weight.grad is not None
    assert agent.actor.weight.grad is not None
    assert agent.actor_logstd.grad is not None


def test_shape_mismatch_fails_closed():
    with pytest.raises(ValueError, match="identical shapes"):
        diagonal_gaussian_kl(
            torch.zeros(2, 2), torch.zeros(2, 2),
            torch.zeros(2, 3), torch.zeros(2, 3),
        )
