"""Learned canonical-view adapter with an immutable V19 control path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_visual_recovery_dual_teacher_ppo import VisualAgent


LAST_CANONICAL_AGENT = None


class CanonicalViewNetwork(nn.Module):
    """Small U-Net that maps an RGB observation into the nominal camera view."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.output = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        image = rgb.permute(0, 3, 1, 2).float().div(255.0)
        first = self.enc1(image)
        second = self.enc2(first)
        latent = self.bottleneck(second)
        up_second = F.interpolate(latent, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up_second = self.dec2(torch.cat((up_second, second), dim=1))
        up_first = F.interpolate(up_second, size=first.shape[-2:], mode="bilinear", align_corners=False)
        residual = torch.tanh(self.output(self.dec1(torch.cat((up_first, first), dim=1))))
        canonical = (image + residual).clamp(0.0, 1.0)
        return canonical.permute(0, 2, 3, 1).mul(255.0)


class PixelDomainRouter(nn.Module):
    """Classify nominal versus shifted frames directly from RGB pixels."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.network(rgb.permute(0, 3, 1, 2).float().div(255.0))


class CanonicalizedV19Agent(nn.Module):
    """Use exact V19 features in-domain and learned canonical pixels OOD."""

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_CANONICAL_AGENT
        self.base = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.canonicalizer = CanonicalViewNetwork()
        self.router = PixelDomainRouter()
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        self.route_threshold = 0.5
        self.route_positive = 0
        self.route_total = 0
        LAST_CANONICAL_AGENT = self

    def initialize_from_v19(self, state: dict) -> None:
        self.base.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def canonicalize(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.canonicalizer(rgb)

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V33 deployment does not use stochastic augmentation")
        base_latent = self.base.encode(rgb)
        canonical_latent = self.base.encode(self.canonicalize(rgb))
        route = torch.sigmoid(self.router(rgb)) >= self.route_threshold
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        return torch.where(route, canonical_latent, base_latent)

    @property
    def learned_route_fraction(self) -> float:
        return self.route_positive / self.route_total if self.route_total else 0.0
