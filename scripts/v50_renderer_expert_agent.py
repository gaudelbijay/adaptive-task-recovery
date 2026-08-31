"""Four-way RGB expert router with dedicated camera and lighting encoders."""

from copy import deepcopy

import torch
import torch.nn as nn

from v47_renderer_expert_agent import RendererExpertV41Agent


class DedicatedRendererExpertAgent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.v47 = RendererExpertV41Agent(*args, **kwargs)
        self.bright_encoder = deepcopy(self.v47.v41.base.encoder)
        self.green_encoder = deepcopy(self.v47.v41.base.encoder)
        self.router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 4),
        )
        self.actor = self.v47.actor
        self.goal_progress_predictor = self.v47.goal_progress_predictor

    def initialize_from_v47(self, state):
        self.v47.load_state_dict(state, strict=True)
        baseline = self.v47.v41.base.encoder.state_dict()
        self.bright_encoder.load_state_dict(baseline, strict=True)
        self.green_encoder.load_state_dict(baseline, strict=True)
        for parameter in self.v47.parameters():
            parameter.requires_grad_(False)

    def router_logits(self, rgb):
        return self.router(rgb.permute(0, 3, 1, 2).float().div(255.0))

    @staticmethod
    def _latent(encoder, rgb):
        return encoder(rgb.permute(0, 3, 1, 2).float().div(255.0))

    def bright_latent(self, rgb):
        return self._latent(self.bright_encoder, rgb)

    def green_latent(self, rgb):
        return self._latent(self.green_encoder, rgb)

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V50 deployment does not use stochastic augmentation")
        choice = self.router_logits(rgb).argmax(dim=1)
        result = self.v47.v41.encode(rgb)
        if bool((choice == 1).any()):
            result = torch.where((choice == 1)[:, None], self.v47.camera_latent(rgb), result)
        if bool((choice == 2).any()):
            result = torch.where((choice == 2)[:, None], self.bright_latent(rgb), result)
        if bool((choice == 3).any()):
            result = torch.where((choice == 3)[:, None], self.green_latent(rgb), result)
        return result
