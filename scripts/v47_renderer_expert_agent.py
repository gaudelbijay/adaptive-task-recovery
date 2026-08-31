"""Deployable RGB-routed experts for geometric, camera, and lighting domains."""

from copy import deepcopy

import torch
import torch.nn as nn

from v41_magnitude_gated_agent import V41MagnitudeGatedDenseV19Agent


class RendererExpertV41Agent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.v41 = V41MagnitudeGatedDenseV19Agent(*args, **kwargs)
        self.camera_encoder = deepcopy(self.v41.base.encoder)
        self.lighting_encoder = deepcopy(self.v41.base.encoder)
        self.router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 3),
        )
        self.actor = self.v41.actor
        self.goal_progress_predictor = self.v41.goal_progress_predictor

    def initialize_from_v45(self, state):
        v41_state = {key.removeprefix("v41."): value for key, value in state.items()
                     if key.startswith("v41.")}
        lighting_state = {
            key.removeprefix("renderer_encoder."): value for key, value in state.items()
            if key.startswith("renderer_encoder.")
        }
        self.v41.load_state_dict(v41_state, strict=True)
        self.camera_encoder.load_state_dict(self.v41.base.encoder.state_dict(), strict=True)
        self.lighting_encoder.load_state_dict(lighting_state, strict=True)
        for parameter in self.v41.parameters():
            parameter.requires_grad_(False)
        for parameter in self.lighting_encoder.parameters():
            parameter.requires_grad_(False)

    def router_logits(self, rgb):
        return self.router(rgb.permute(0, 3, 1, 2).float().div(255.0))

    @staticmethod
    def _expert_latent(encoder, rgb):
        return encoder(rgb.permute(0, 3, 1, 2).float().div(255.0))

    def camera_latent(self, rgb):
        return self._expert_latent(self.camera_encoder, rgb)

    def lighting_latent(self, rgb):
        return self._expert_latent(self.lighting_encoder, rgb)

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V47 deployment does not use stochastic augmentation")
        choice = self.router_logits(rgb).argmax(dim=1)
        baseline = self.v41.encode(rgb)
        result = baseline
        if bool((choice == 1).any()):
            camera = self.camera_latent(rgb)
            result = torch.where((choice == 1)[:, None], camera, result)
        if bool((choice == 2).any()):
            lighting = self.lighting_latent(rgb)
            result = torch.where((choice == 2)[:, None], lighting, result)
        return result
