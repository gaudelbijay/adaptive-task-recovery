"""RGB-only V19-preserving hybrid used by the V32 robustness experiment."""

from __future__ import annotations

import torch
import torch.nn as nn

from train_visual_recovery_dual_teacher_ppo import VisualAgent


LAST_HYBRID_AGENT = None


class HybridDomainAgent(nn.Module):
    """Route pixels between an immutable V19 path and a robust visual adapter.

    The evaluator supplies no domain identifier.  A classifier over frozen V19
    visual features makes the routing decision from RGB alone.  The actor and
    learned progress head are shared and initialized from V19, so a correctly
    routed in-domain frame executes the exact incumbent computation.
    """

    def __init__(
        self, image_size, proprio_dim, critic_dim, action_dim, asymmetric,
        aug_pad, privileged_aux_dim=0, learned_goal_progress=False,
    ):
        super().__init__()
        global LAST_HYBRID_AGENT
        self.base = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.robust = VisualAgent(
            image_size, proprio_dim, critic_dim, action_dim, asymmetric,
            aug_pad, privileged_aux_dim, learned_goal_progress,
        )
        self.router = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        # The shared evaluator intentionally accesses these two attributes.
        # Aliasing the immutable base modules keeps its causal progress-head
        # intervention exact without exposing any domain label.
        self.actor = self.base.actor
        self.goal_progress_predictor = self.base.goal_progress_predictor
        self.route_threshold = 0.5
        self.route_positive = 0
        self.route_total = 0
        LAST_HYBRID_AGENT = self

    def initialize_from_v19(self, state: dict) -> None:
        self.base.load_state_dict(state, strict=True)
        self.robust.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.robust.parameters():
            parameter.requires_grad_(False)
        for parameter in self.robust.encoder.parameters():
            parameter.requires_grad_(True)
        if self.robust.privileged_predictor is None:
            raise ValueError("V32 requires the geometry prediction head")
        for parameter in self.robust.privileged_predictor.parameters():
            parameter.requires_grad_(True)

    def route_probability(self, rgb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_latent = self.base.encode(rgb)
        return torch.sigmoid(self.router(base_latent))

    def robust_action(self, rgb: torch.Tensor, proprio: torch.Tensor):
        latent = self.robust.encode(rgb)
        parts = [latent, proprio]
        if self.goal_progress_predictor is not None:
            parts.append(torch.sigmoid(self.goal_progress_predictor(latent)))
        return torch.tanh(self.actor(torch.cat(parts, dim=1))), latent

    def encode(self, rgb: torch.Tensor, augment: bool = False) -> torch.Tensor:
        if augment:
            raise ValueError("V32 deployment routing does not use stochastic augmentation")
        base_latent = self.base.encode(rgb)
        robust_latent = self.robust.encode(rgb)
        probability = torch.sigmoid(self.router(base_latent))
        route = probability >= self.route_threshold
        self.route_positive += int(route.sum().detach())
        self.route_total += int(route.numel())
        return torch.where(route, robust_latent, base_latent)

    @property
    def learned_route_fraction(self) -> float:
        return self.route_positive / self.route_total if self.route_total else 0.0

