"""Clean-preserving multi-view feature adapter over frozen V41 control."""

from copy import deepcopy

import torch
import torch.nn as nn

from v41_magnitude_gated_agent import V41MagnitudeGatedDenseV19Agent


class MultiViewFeatureV41Agent(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.v41 = V41MagnitudeGatedDenseV19Agent(*args, **kwargs)
        self.renderer_encoder = deepcopy(self.v41.base.encoder)
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, -4.0)
        self.route_threshold = 0.9
        self.actor = self.v41.actor
        self.goal_progress_predictor = self.v41.goal_progress_predictor
        self.route_positive = 0
        self.route_total = 0

    def initialize_from_v40(self, state):
        self.v41.load_state_dict(state, strict=True)
        self.renderer_encoder.load_state_dict(self.v41.base.encoder.state_dict(), strict=True)
        for parameter in self.v41.parameters():
            parameter.requires_grad_(False)

    def route_logits(self, rgb):
        return self.router(rgb.permute(0, 3, 1, 2).float().div(255.0)).squeeze(1)

    def renderer_latent(self, rgb):
        return self.renderer_encoder(rgb.permute(0, 3, 1, 2).float().div(255.0))

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V44 deployment does not use stochastic augmentation")
        logits = self.route_logits(rgb)
        route = torch.sigmoid(logits) >= self.route_threshold
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        baseline = self.v41.encode(rgb)
        if not bool(route.any()):
            return baseline
        adapted = self.renderer_latent(rgb)
        return torch.where(route[:, None], adapted, baseline)

    @property
    def learned_route_fraction(self):
        return self.route_positive / self.route_total if self.route_total else 0.0


class AlwaysMultiViewFeatureV41Agent(MultiViewFeatureV41Agent):
    """Use the invariant encoder on every frame; training must protect clean features."""

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V45 deployment does not use stochastic augmentation")
        self.route_positive += int(rgb.shape[0]); self.route_total += int(rgb.shape[0])
        return self.renderer_latent(rgb)


class CalibratedHybridFeatureV41Agent(MultiViewFeatureV41Agent):
    """Route between V41 and the clean-anchored encoder at a frozen threshold."""

    deployment_route_threshold = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_threshold = float(self.deployment_route_threshold)
