"""Use exact V39 geometry control unless V53 confidently selects a renderer expert."""

import torch

from v58_hierarchical_geometry_agent import HierarchicalGeometryCompositionAgent


class RendererV39CompositionAgent(HierarchicalGeometryCompositionAgent):
    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V59 deployment does not use stochastic augmentation")
        geometry_result = self.detector.encode(rgb)
        renderer_result = self.base.encode(rgb)
        probability = torch.softmax(self.base.router_logits(rgb), dim=1)
        confidence, choice = probability.max(dim=1)
        route = (choice != 0) & (confidence >= self.base.route_confidence)
        return torch.where(route[:, None], renderer_result, geometry_result)
