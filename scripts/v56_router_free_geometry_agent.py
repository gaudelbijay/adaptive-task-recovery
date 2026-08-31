"""Renderer-first V53 with V54 joint geometry correction as the default."""

import torch

from v54_continuous_geometry_agent import ContinuousGeometryCompositionAgent


class RouterFreeGeometryCompositionAgent(ContinuousGeometryCompositionAgent):
    def encode(self, rgb, augment=False):
        if augment: raise ValueError("V56 deployment does not use stochastic augmentation")
        renderer_result = self.base.encode(rgb)
        probability = torch.softmax(self.base.router_logits(rgb), dim=1)
        confidence, choice = probability.max(dim=1)
        renderer_route = (choice != 0) & (confidence >= self.base.route_confidence)
        geometry_result = self.geometry_latent(3, rgb)
        return torch.where(renderer_route[:, None], renderer_result, geometry_result)
