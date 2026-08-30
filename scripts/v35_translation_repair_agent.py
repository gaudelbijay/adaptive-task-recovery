"""Learned global-translation repair in front of a frozen V34 policy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from v34_factorized_canonical_agent import FactorizedCanonicalV19Agent


LAST_TRANSLATION_AGENT = None


def sample_warp(rgb: torch.Tensor, sampling_offset_pixels: torch.Tensor) -> torch.Tensor:
    """Sample each image at (x+dx,y+dy), using border padding."""
    image = rgb.permute(0, 3, 1, 2).float().div(255.0)
    batch, _, height, width = image.shape
    theta = torch.eye(2, 3, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(batch, 1, 1)
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    offsets = torch.stack(
        (
            sampling_offset_pixels[:, 0] * (2.0 / width),
            sampling_offset_pixels[:, 1] * (2.0 / height),
        ),
        dim=-1,
    ).view(batch, 1, 1, 2)
    warped = F.grid_sample(
        image, grid + offsets, mode="bilinear", padding_mode="border",
        align_corners=False,
    )
    return warped.permute(0, 2, 3, 1).mul(255.0)


def synthesize_content_translation(rgb: torch.Tensor, content_shift_pixels: torch.Tensor) -> torch.Tensor:
    """Move image content by (+right,+down) pixels."""
    return sample_warp(rgb, -content_shift_pixels)


class TranslationEstimator(nn.Module):
    def __init__(self, max_shift_pixels: float = 8.0):
        super().__init__()
        self.max_shift_pixels = float(max_shift_pixels)
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 96), nn.ReLU(),
        )
        self.shift_logit = nn.Linear(96, 1)
        self.offset = nn.Linear(96, 2)

    def forward(self, rgb: torch.Tensor):
        features = self.features(rgb.permute(0, 3, 1, 2).float().div(255.0))
        return self.shift_logit(features), torch.tanh(self.offset(features)) * self.max_shift_pixels


class TranslationRepairedV34Agent(nn.Module):
    """Correct learned global translations, then execute the frozen V34 policy."""

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_TRANSLATION_AGENT
        self.robust = FactorizedCanonicalV19Agent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.translation = TranslationEstimator()
        self.actor = self.robust.actor
        self.goal_progress_predictor = self.robust.goal_progress_predictor
        self.translation_positive = 0
        self.translation_total = 0
        self.offset_sum = 0.0
        LAST_TRANSLATION_AGENT = self

    def initialize_from_v34(self, state: dict) -> None:
        self.robust.load_state_dict(state, strict=True)
        for parameter in self.robust.parameters():
            parameter.requires_grad_(False)

    def correct_translation(self, rgb: torch.Tensor, hard_route: bool = True):
        logits, offsets = self.translation(rgb)
        corrected = sample_warp(rgb, offsets)
        if hard_route:
            route = torch.sigmoid(logits) >= 0.5
            corrected = torch.where(route.view(-1, 1, 1, 1), corrected, rgb.float())
            self.translation_positive += int(route.sum().detach())
            self.translation_total += int(route.numel())
            self.offset_sum += float((offsets.norm(dim=1) * route.squeeze(1)).sum().detach())
        return corrected, logits, offsets

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V35 deployment does not use stochastic augmentation")
        corrected, _, _ = self.correct_translation(rgb, hard_route=True)
        return self.robust.encode(corrected)

    @property
    def learned_translation_route_fraction(self) -> float:
        return self.translation_positive / self.translation_total if self.translation_total else 0.0

    @property
    def mean_routed_translation_magnitude(self) -> float:
        return self.offset_sum / self.translation_positive if self.translation_positive else 0.0
