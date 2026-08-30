"""Factorized learned spatial/photometric canonicalization for frozen V19."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_visual_recovery_dual_teacher_ppo import VisualAgent


LAST_FACTORIZED_AGENT = None


class FactorizedCanonicalizer(nn.Module):
    """Predict a dense spatial warp, then a bounded photometric residual."""

    def __init__(self, max_flow_pixels: float = 8.0):
        super().__init__()
        self.max_flow_pixels = float(max_flow_pixels)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1), nn.ReLU(),
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 48, 3, padding=1), nn.ReLU(),
        )
        self.flow_head = nn.Conv2d(48, 2, 3, padding=1)
        self.residual_head = nn.Conv2d(51, 3, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, rgb: torch.Tensor, return_flow: bool = False):
        image = rgb.permute(0, 3, 1, 2).float().div(255.0)
        first = self.enc1(image)
        second = self.enc2(first)
        latent = self.enc3(second)
        up_second = F.interpolate(latent, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up_second = self.dec2(torch.cat((up_second, second), dim=1))
        up_first = F.interpolate(up_second, size=first.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.dec1(torch.cat((up_first, first), dim=1))
        flow_pixels = torch.tanh(self.flow_head(decoded)) * self.max_flow_pixels

        batch, _, height, width = image.shape
        theta = torch.eye(2, 3, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(batch, 1, 1)
        grid = F.affine_grid(theta, image.shape, align_corners=False)
        flow_grid = torch.stack(
            (flow_pixels[:, 0] * (2.0 / width), flow_pixels[:, 1] * (2.0 / height)),
            dim=-1,
        )
        warped = F.grid_sample(
            image, grid + flow_grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )
        residual = torch.tanh(self.residual_head(torch.cat((decoded, warped), dim=1)))
        canonical = (warped + residual).clamp(0.0, 1.0)
        result = canonical.permute(0, 2, 3, 1).mul(255.0)
        return (result, flow_pixels) if return_flow else result


class MulticlassPixelRouter(nn.Module):
    """Classify nominal and seven observed RGB domains without evaluator labels."""

    def __init__(self, domain_count: int = 8):
        super().__init__()
        self.domain_count = int(domain_count)
        self.network = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(96, 64), nn.ReLU(), nn.Linear(64, self.domain_count),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.network(rgb.permute(0, 3, 1, 2).float().div(255.0))


class FactorizedCanonicalV19Agent(nn.Module):
    """Preserve exact V19 for predicted nominal frames; canonicalize all others."""

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_FACTORIZED_AGENT
        self.base = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.canonicalizer = FactorizedCanonicalizer()
        self.router = MulticlassPixelRouter(domain_count=8)
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        self.route_positive = 0
        self.route_total = 0
        LAST_FACTORIZED_AGENT = self

    def initialize_from_v19(self, state: dict) -> None:
        self.base.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def canonicalize(self, rgb: torch.Tensor, return_flow: bool = False):
        return self.canonicalizer(rgb, return_flow=return_flow)

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V34 deployment does not use stochastic augmentation")
        base_latent = self.base.encode(rgb)
        canonical_latent = self.base.encode(self.canonicalize(rgb))
        route = self.router(rgb).argmax(dim=1, keepdim=True) != 0
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        return torch.where(route, canonical_latent, base_latent)

    @property
    def learned_route_fraction(self) -> float:
        return self.route_positive / self.route_total if self.route_total else 0.0
