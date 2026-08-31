"""Binary-routed geometry composition using V54's joint-transform expert."""

import torch
import torch.nn as nn

from v54_continuous_geometry_agent import ContinuousGeometryCompositionAgent


class BinaryGeometryCompositionAgent(ContinuousGeometryCompositionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 2),
        )

    def encode(self, rgb, augment=False):
        if augment: raise ValueError("V55 deployment does not use stochastic augmentation")
        result = self.base.encode(rgb); probability = torch.softmax(self.router_logits(rgb), dim=1)[:, 1]
        route = probability >= self.route_confidence
        if bool(route.any()): result = torch.where(route[:, None], self.geometry_latent(3, rgb), result)
        return result
