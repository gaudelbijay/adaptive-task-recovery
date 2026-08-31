"""Hierarchical geometry expert beneath the frozen V47 renderer router."""

from copy import deepcopy

import torch
import torch.nn as nn

from v47_renderer_expert_agent import RendererExpertV41Agent


class HierarchicalGeometryExpertAgent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.v47 = RendererExpertV41Agent(*args, **kwargs)
        self.geometry_encoder = deepcopy(self.v47.v41.base.encoder)
        self.geometry_router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 1),
        )
        self.actor = self.v47.actor
        self.goal_progress_predictor = self.v47.goal_progress_predictor
        self.geometry_threshold = 0.5

    def initialize_from_v47(self, state):
        self.v47.load_state_dict(state, strict=True)
        self.geometry_encoder.load_state_dict(self.v47.v41.base.encoder.state_dict(), strict=True)
        for parameter in self.v47.parameters():
            parameter.requires_grad_(False)

    def geometry_logits(self, rgb):
        return self.geometry_router(rgb.permute(0, 3, 1, 2).float().div(255.0)).squeeze(1)

    def geometry_latent(self, rgb):
        return self.geometry_encoder(rgb.permute(0, 3, 1, 2).float().div(255.0))

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V48 deployment does not use stochastic augmentation")
        top_choice = self.v47.router_logits(rgb).argmax(dim=1)
        result = self.v47.v41.encode(rgb)
        geometry_choice = (top_choice == 0) & (
            torch.sigmoid(self.geometry_logits(rgb)) >= self.geometry_threshold
        )
        if bool(geometry_choice.any()):
            geometry = self.geometry_latent(rgb)
            result = torch.where(geometry_choice[:, None], geometry, result)
        if bool((top_choice == 1).any()):
            camera = self.v47.camera_latent(rgb)
            result = torch.where((top_choice == 1)[:, None], camera, result)
        if bool((top_choice == 2).any()):
            lighting = self.v47.lighting_latent(rgb)
            result = torch.where((top_choice == 2)[:, None], lighting, result)
        return result
