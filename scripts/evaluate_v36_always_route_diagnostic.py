#!/usr/bin/env python3
"""Development-only V36 diagnostic that always applies the learned correction."""

import torch

import evaluate_visual_recovery_ppo as base
import v36_continuous_canonical_agent as v36
from evaluate_v35_visual_recovery_unseen_ood import install_extensions
from evaluate_v36_visual_recovery import install_protocol_adapter


class AlwaysRouteAgent(v36.ContinuousCanonicalV19Agent):
    """Remove only the hard router to diagnose correction versus detection."""

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V36 diagnostic does not use stochastic augmentation")
        corrected, _, _, _, _ = self.canonicalizer.correct(rgb, hard_route=False)
        self.route_positive += int(rgb.shape[0])
        self.route_total += int(rgb.shape[0])
        return self.base.encode(corrected)


if __name__ == "__main__":
    install_extensions()
    v36.ContinuousCanonicalV19Agent = AlwaysRouteAgent
    install_protocol_adapter()
    base.main()
