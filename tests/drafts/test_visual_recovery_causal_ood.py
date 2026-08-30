import pytest
import sys
from pathlib import Path

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_visual_recovery_ppo import (  # noqa: E402
    VisualAgent,
    apply_visual_perturbation,
    deterministic_action_with_progress_mode,
)
from atr.envs.learned_recovery_v3_ood import (  # noqa: E402
    PROFILES, camera_eye, lighting_parameters,
)


class TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.goal_progress_predictor = torch.nn.Linear(3, 2, bias=False)
        self.actor = torch.nn.Linear(3 + 2 + 2, 1, bias=False)
        with torch.no_grad():
            self.goal_progress_predictor.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            ]))
            self.actor.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]]))

    def encode(self, rgb):
        return rgb.float().mean(dim=(1, 2)) / 255.0


def test_progress_interventions_change_only_the_actor_head_input():
    agent = TinyAgent()
    rgb = torch.tensor([
        [[[255, 0, 0]]], [[[0, 255, 0]]], [[[0, 0, 255]]],
    ], dtype=torch.uint8)
    proprio = torch.zeros((3, 2))
    normal, predicted = deterministic_action_with_progress_mode(agent, rgb, proprio, "normal")
    shifted, shifted_predicted = deterministic_action_with_progress_mode(
        agent, rgb, proprio, "cyclic_shift",
    )
    zero, _ = deterministic_action_with_progress_mode(agent, rgb, proprio, "zero")
    one, _ = deterministic_action_with_progress_mode(agent, rgb, proprio, "one")
    assert torch.equal(predicted, shifted_predicted)
    assert not torch.equal(normal, shifted)
    assert torch.equal(zero, one)


def test_normal_mode_is_exactly_the_existing_deterministic_actor_path():
    torch.manual_seed(4)
    agent = VisualAgent(
        64, proprio_dim=5, critic_dim=0, action_dim=3, asymmetric=False,
        aug_pad=0, learned_goal_progress=True,
    ).eval()
    rgb = torch.randint(0, 256, (4, 64, 64, 3), dtype=torch.uint8)
    proprio = torch.randn(4, 5)
    expected = agent.get_action(rgb, proprio, deterministic=True)
    actual, _ = deterministic_action_with_progress_mode(
        agent, rgb, proprio, "normal",
    )
    assert torch.equal(actual, expected)


def test_sensor_perturbations_are_deterministic_shape_preserving_and_nontrivial():
    rgb = torch.arange(2 * 8 * 8 * 3, dtype=torch.uint8).reshape(2, 8, 8, 3)
    assert apply_visual_perturbation(rgb, "none") is rgb
    for mode in ("pixel_shift_right_4", "brightness_70", "warm_color_shift"):
        first = apply_visual_perturbation(rgb, mode)
        second = apply_visual_perturbation(rgb, mode)
        assert first.shape == rgb.shape
        assert first.dtype == rgb.dtype
        assert torch.equal(first, second)
        assert not torch.equal(first, rgb)


def test_nonexistent_progress_head_fails_closed_for_interventions():
    agent = TinyAgent()
    agent.goal_progress_predictor = None
    rgb = torch.zeros((2, 1, 1, 3), dtype=torch.uint8)
    proprio = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="requires a learned head"):
        deterministic_action_with_progress_mode(agent, rgb, proprio, "zero")


def test_rendered_ood_profiles_are_frozen_and_distinct():
    assert PROFILES == (
        "camera_left_5cm", "camera_high_5cm", "lighting_dim", "lighting_warm",
    )
    nominal = torch.tensor([0.45, 0.0, 0.72], dtype=torch.double)
    assert torch.allclose(torch.from_numpy(camera_eye("camera_left_5cm")), nominal + torch.tensor([0.0, 0.05, 0.0]))
    assert torch.allclose(torch.from_numpy(camera_eye("camera_high_5cm")), nominal + torch.tensor([0.0, 0.0, 0.05]))
    assert lighting_parameters("camera_left_5cm") is None
    assert lighting_parameters("lighting_dim") != lighting_parameters("lighting_warm")
