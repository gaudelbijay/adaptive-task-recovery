"""Continuous similarity/photometric canonicalization ahead of frozen V19."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_visual_recovery_dual_teacher_ppo import VisualAgent


LAST_V36_AGENT = None


def similarity_theta(parameters: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert pixel translation, radians, and log-scale to sampling matrices."""

    tx, ty, angle, log_scale = parameters.unbind(dim=1)
    scale = log_scale.exp()
    cosine = angle.cos() * scale
    sine = angle.sin() * scale
    theta = torch.zeros((parameters.shape[0], 2, 3), device=parameters.device, dtype=parameters.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    theta[:, 0, 2] = tx * (2.0 / width)
    theta[:, 1, 2] = ty * (2.0 / height)
    return theta


def invert_theta(theta: torch.Tensor) -> torch.Tensor:
    bottom = torch.tensor([0.0, 0.0, 1.0], device=theta.device, dtype=theta.dtype)
    homogeneous = torch.cat((theta, bottom.view(1, 1, 3).expand(theta.shape[0], -1, -1)), dim=1)
    return torch.linalg.inv(homogeneous)[:, :2]


def sample_affine(rgb: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2).float().div(255.0)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    result = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=False,
    )
    return result.permute(0, 2, 3, 1).mul(255.0)


def synthesize_corruption(
    rgb: torch.Tensor,
    correction_parameters: torch.Tensor,
    correction_gain: torch.Tensor,
    correction_bias: torch.Tensor,
) -> torch.Tensor:
    """Generate an image whose exact inverse correction is supplied."""

    height, width = rgb.shape[1:3]
    correction = similarity_theta(correction_parameters, height, width)
    corrupted = sample_affine(rgb, invert_theta(correction)).div(255.0)
    corrupted = (corrupted - correction_bias[:, None, None, :]) / correction_gain[:, None, None, :]
    return corrupted.clamp(0.0, 1.0).mul(255.0)


class ContinuousCanonicalizer(nn.Module):
    """Position-aware estimator for continuous geometry and color correction."""

    def __init__(
        self, max_translation: float = 10.0, max_rotation_degrees: float = 10.0,
        max_log_scale: float = math.log(1.2), route_threshold: float = 0.9,
    ):
        super().__init__()
        self.max_translation = float(max_translation)
        self.max_rotation = math.radians(float(max_rotation_degrees))
        self.max_log_scale = float(max_log_scale)
        self.route_threshold = float(route_threshold)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
        )
        self.route_logit = nn.Linear(128, 1)
        self.geometry = nn.Linear(128, 4)
        self.photometric = nn.Linear(128, 6)
        nn.init.zeros_(self.geometry.weight)
        nn.init.zeros_(self.geometry.bias)
        nn.init.zeros_(self.photometric.weight)
        nn.init.zeros_(self.photometric.bias)
        nn.init.zeros_(self.route_logit.weight)
        nn.init.constant_(self.route_logit.bias, -3.0)

    def estimate(self, rgb: torch.Tensor):
        features = self.features(rgb.permute(0, 3, 1, 2).float().div(255.0))
        raw_geometry = torch.tanh(self.geometry(features))
        parameters = torch.stack((
            raw_geometry[:, 0] * self.max_translation,
            raw_geometry[:, 1] * self.max_translation,
            raw_geometry[:, 2] * self.max_rotation,
            raw_geometry[:, 3] * self.max_log_scale,
        ), dim=1)
        color = torch.tanh(self.photometric(features))
        gain = torch.exp(color[:, :3] * math.log(1.6))
        bias = color[:, 3:] * 0.25
        return self.route_logit(features), parameters, gain, bias

    def correct(self, rgb: torch.Tensor, hard_route: bool = True):
        logits, parameters, gain, bias = self.estimate(rgb)
        theta = similarity_theta(parameters, rgb.shape[1], rgb.shape[2])
        corrected = sample_affine(rgb, theta).div(255.0)
        corrected = (corrected * gain[:, None, None, :] + bias[:, None, None, :]).clamp(0.0, 1.0)
        corrected = corrected.mul(255.0)
        probability = torch.sigmoid(logits)
        if hard_route:
            route = probability >= self.route_threshold
            corrected = torch.where(route.view(-1, 1, 1, 1), corrected, rgb.float())
        return corrected, logits, parameters, gain, bias


class ContinuousCanonicalV19Agent(nn.Module):
    """Preserve exact V19 unless a position-aware RGB correction is confident."""

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_V36_AGENT
        self.base = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.canonicalizer = ContinuousCanonicalizer()
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        self.route_positive = 0
        self.route_total = 0
        LAST_V36_AGENT = self

    def initialize_from_v19(self, state: dict) -> None:
        self.base.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V36 deployment does not use stochastic augmentation")
        corrected, logits, _, _, _ = self.canonicalizer.correct(rgb, hard_route=True)
        route = torch.sigmoid(logits) >= self.canonicalizer.route_threshold
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        return self.base.encode(corrected)

    @property
    def learned_route_fraction(self) -> float:
        return self.route_positive / self.route_total if self.route_total else 0.0
