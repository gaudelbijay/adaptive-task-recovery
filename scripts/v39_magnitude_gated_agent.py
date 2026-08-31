"""Magnitude-gated V39 deployment agent."""

import torch

from v37_dense_canonical_agent import DenseCanonicalV19Agent


class MagnitudeGatedDenseV19Agent(DenseCanonicalV19Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnitude_threshold = 0.003
        self.route_positive = 0
        self.route_total = 0

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V39 deployment does not use stochastic augmentation")
        corrected = self.correct(rgb)
        magnitude = (corrected - rgb.float()).abs().mean(dim=(1, 2, 3)).div(255.0)
        route = magnitude >= self.magnitude_threshold
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        selected = torch.where(route.view(-1, 1, 1, 1), corrected, rgb.float())
        return self.base.encode(selected)

    @property
    def learned_route_fraction(self):
        return self.route_positive / self.route_total if self.route_total else 0.0
