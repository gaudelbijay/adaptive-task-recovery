"""V53 renderer routing, V39 geometry detection, and V54 specialist selection."""

import torch

from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent
from v54_continuous_geometry_agent import ContinuousGeometryCompositionAgent


class HierarchicalGeometryCompositionAgent(ContinuousGeometryCompositionAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector = MagnitudeGatedDenseV19Agent(*args, **kwargs)
        self.geometry_detection_threshold = 0.003

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V58 deployment does not use stochastic augmentation")
        result = self.base.encode(rgb)
        renderer_probability = torch.softmax(self.base.router_logits(rgb), dim=1)
        renderer_confidence, renderer_choice = renderer_probability.max(dim=1)
        renderer_route = (renderer_choice != 0) & (
            renderer_confidence >= self.base.route_confidence
        )
        detected_correction = self.detector.correct(rgb)
        magnitude = (detected_correction - rgb.float()).abs().mean((1, 2, 3)).div(255.0)
        geometry_route = (magnitude >= self.geometry_detection_threshold) & ~renderer_route
        # The upstream detector has already established that geometry changed,
        # so class 0 is no longer a valid category. V54 only selects which of
        # its four independently trained correction experts to execute.
        geometry_choice = self.router_logits(rgb)[:, 1:].argmax(dim=1)
        for index in range(4):
            route = geometry_route & (geometry_choice == index)
            if bool(route.any()):
                result = torch.where(route[:, None], self.geometry_latent(index, rgb), result)
        return result
