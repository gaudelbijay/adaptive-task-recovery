"""Always-on global plus dense RGB canonicalization ahead of frozen V19."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from train_visual_recovery_dual_teacher_ppo import VisualAgent
from v36_continuous_canonical_agent import ContinuousCanonicalizer


LAST_V37_AGENT = None


class DenseCanonicalV19Agent(nn.Module):
    """Correct every frame while an identity loss protects the clean domain."""

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_V37_AGENT
        self.base = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.global_canonicalizer = ContinuousCanonicalizer(
            max_translation=8.0, max_rotation_degrees=8.0,
            max_log_scale=max(abs(math.log(0.88)), abs(math.log(1.12))),
        )
        self.dense_residual = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Tanh(),
        )
        nn.init.zeros_(self.dense_residual[-2].weight)
        nn.init.zeros_(self.dense_residual[-2].bias)
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        LAST_V37_AGENT = self

    def initialize_from_v36(self, state: dict) -> None:
        base_state = {key.removeprefix("base."): value for key, value in state.items() if key.startswith("base.")}
        canonical_state = {
            key.removeprefix("canonicalizer."): value
            for key, value in state.items() if key.startswith("canonicalizer.")
        }
        self.base.load_state_dict(base_state, strict=True)
        self.global_canonicalizer.load_state_dict(canonical_state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def correct(self, rgb: torch.Tensor) -> torch.Tensor:
        globally_corrected, _, _, _, _ = self.global_canonicalizer.correct(rgb, hard_route=False)
        normalized = globally_corrected.permute(0, 3, 1, 2).float().div(255.0)
        residual = 0.25 * self.dense_residual(normalized)
        return (normalized + residual).clamp(0.0, 1.0).permute(0, 2, 3, 1).mul(255.0)

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V37 deployment does not use stochastic augmentation")
        return self.base.encode(self.correct(rgb))
