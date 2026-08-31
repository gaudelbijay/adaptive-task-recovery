"""V53 renderer recovery plus confidence-gated continuous geometry experts."""

from copy import deepcopy
import math

import torch
import torch.nn as nn

from v53_opened_renderer_agent import OpenedRendererExpertAgent
from v39_magnitude_gated_agent import MagnitudeGatedDenseV19Agent


class ContinuousGeometryCompositionAgent(nn.Module):
    """Keep V53 exact unless a learned RGB router selects a geometry expert."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.base = OpenedRendererExpertAgent(*args, **kwargs)
        template = MagnitudeGatedDenseV19Agent(*args, **kwargs)
        self.geometry_encoder = template.base
        self.global_correctors = nn.ModuleList(
            [deepcopy(template.global_canonicalizer) for _ in range(4)]
        )
        self.dense_correctors = nn.ModuleList(
            [deepcopy(template.dense_residual) for _ in range(4)]
        )
        self.router = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 5),
        )
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        self.route_confidence = 0.9

    def initialize(self, v53_state, v39_state):
        self.base.load_state_dict(v53_state, strict=True)
        geometry_state = {
            key.removeprefix("base."): value
            for key, value in v39_state.items() if key.startswith("base.")
        }
        global_state = {
            key.removeprefix("global_canonicalizer."): value
            for key, value in v39_state.items() if key.startswith("global_canonicalizer.")
        }
        dense_state = {
            key.removeprefix("dense_residual."): value
            for key, value in v39_state.items() if key.startswith("dense_residual.")
        }
        self.geometry_encoder.load_state_dict(geometry_state, strict=True)
        for module in self.global_correctors:
            module.load_state_dict(global_state, strict=True)
            module.max_translation = 6.0
            module.max_rotation = math.radians(10.0)
            module.max_log_scale = abs(math.log(0.84))
        for module in self.dense_correctors:
            module.load_state_dict(dense_state, strict=True)
        # The two independently audited branches must share the deployed control head.
        own = self.actor.state_dict()
        other = self.geometry_encoder.actor.state_dict()
        if own.keys() != other.keys() or any(not torch.equal(own[k], other[k]) for k in own):
            raise ValueError("V53 and V39 actor tensors differ")
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.geometry_encoder.parameters():
            parameter.requires_grad_(False)

    def router_logits(self, rgb):
        return self.router(rgb.permute(0, 3, 1, 2).float().div(255.0))

    def correct(self, index, rgb):
        corrected, *_ = self.global_correctors[index].correct(rgb, hard_route=False)
        normalized = corrected.permute(0, 3, 1, 2).float().div(255.0)
        residual = 0.25 * self.dense_correctors[index](normalized)
        return (normalized + residual).clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0)

    def geometry_latent(self, index, rgb):
        return self.geometry_encoder.encode(self.correct(index, rgb))

    def encode(self, rgb, augment=False):
        if augment:
            raise ValueError("V54 deployment does not use stochastic augmentation")
        result = self.base.encode(rgb)
        probability = torch.softmax(self.router_logits(rgb), dim=1)
        confidence, choice = probability.max(dim=1)
        for label in range(1, 5):
            route = (choice == label) & (confidence >= self.route_confidence)
            if bool(route.any()):
                result = torch.where(route[:, None], self.geometry_latent(label - 1, rgb), result)
        return result
