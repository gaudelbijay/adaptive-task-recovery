#!/usr/bin/env python3
"""Development-only V38 diagnostic with an RGB correction-magnitude fallback."""

import os

import torch

import evaluate_visual_recovery_ppo as base
import v37_dense_canonical_agent as dense
from evaluate_v35_visual_recovery_unseen_ood import install_extensions
from evaluate_v38_visual_recovery import install_protocol_adapter


class MagnitudeGatedAgent(dense.DenseCanonicalV19Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magnitude_threshold = float(os.environ["ATR_V38_MAGNITUDE_THRESHOLD"])

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V38 diagnostic does not use stochastic augmentation")
        corrected = self.correct(rgb)
        magnitude = (corrected - rgb.float()).abs().mean(dim=(1, 2, 3)).div(255.0)
        route = magnitude >= self.magnitude_threshold
        selected = torch.where(route.view(-1, 1, 1, 1), corrected, rgb.float())
        return self.base.encode(selected)


if __name__ == "__main__":
    install_extensions()
    dense.DenseCanonicalV19Agent = MagnitudeGatedAgent
    install_protocol_adapter()
    base.main()
