"""Hierarchical V51 controller with a V43 subpixel-only specialist."""

import torch
import torch.nn as nn

from v41_magnitude_gated_agent import V41MagnitudeGatedDenseV19Agent
from v51_hierarchical_renderer_agent import HierarchicalRendererExpertAgent


class SubpixelSpecialistAgent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.v51 = HierarchicalRendererExpertAgent(*args, **kwargs)
        self.subpixel = V41MagnitudeGatedDenseV19Agent(*args, **kwargs)
        self.subpixel_router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1),
        )
        self.actor = self.v51.actor
        self.goal_progress_predictor = self.v51.goal_progress_predictor
        self.subpixel_threshold = 0.5

    def initialize_sources(self, v51_state, v43_state):
        self.v51.load_state_dict(v51_state, strict=True)
        self.subpixel.load_state_dict(v43_state, strict=True)
        for module in (self.v51, self.subpixel):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def subpixel_logits(self, rgb):
        return self.subpixel_router(rgb.permute(0, 3, 1, 2).float().div(255.0)).squeeze(1)

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V52 deployment does not use stochastic augmentation")
        result = self.v51.encode(rgb)
        top_choice = self.v51.v47.router_logits(rgb).argmax(dim=1)
        route = (top_choice == 0) & (
            torch.sigmoid(self.subpixel_logits(rgb)) >= self.subpixel_threshold
        )
        if bool(route.any()):
            specialist = self.subpixel.encode(rgb)
            result = torch.where(route[:, None], specialist, result)
        return result
