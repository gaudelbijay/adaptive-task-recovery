"""V59 with a V54 joint specialist applied only after V39 correction."""

import torch

from v59_renderer_v39_agent import RendererV39CompositionAgent


class JointResidualCompositionAgent(RendererV39CompositionAgent):
    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V60 deployment does not use stochastic augmentation")
        result = self.detector.encode(rgb)
        renderer_result = self.base.encode(rgb)
        renderer_probability = torch.softmax(self.base.router_logits(rgb), dim=1)
        renderer_confidence, renderer_choice = renderer_probability.max(dim=1)
        renderer_route = (renderer_choice != 0) & (
            renderer_confidence >= self.base.route_confidence
        )
        result = torch.where(renderer_route[:, None], renderer_result, result)

        first_correction = self.detector.correct(rgb)
        magnitude = (first_correction - rgb.float()).abs().mean((1, 2, 3)).div(255.0)
        geometry_route = (magnitude >= self.geometry_detection_threshold) & ~renderer_route
        geometry_choice = self.router_logits(rgb)[:, 1:].argmax(dim=1)
        joint_route = geometry_route & (geometry_choice == 3)
        if bool(joint_route.any()):
            residual_corrected = self.correct(3, first_correction)
            residual_latent = self.geometry_encoder.encode(residual_corrected)
            result = torch.where(joint_route[:, None], residual_latent, result)
        return result
