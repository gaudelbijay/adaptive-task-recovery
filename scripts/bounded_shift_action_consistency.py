"""Bounded random-shift consistency for continuous-control policies."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bounded_action_mean(agent, rgb, proprio, *, augment: bool) -> torch.Tensor:
    """Return the deterministic action in the environment's bounded space."""

    latent = agent.encode(rgb, augment=augment)
    actor_parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        actor_parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    return torch.tanh(agent.actor(torch.cat(actor_parts, dim=1)))


def bounded_shift_action_consistency(agent, rgb, proprio) -> torch.Tensor:
    """Huber consistency from stopped clean actions to shifted-image actions.

    Unlike Gaussian KL, this loss is evaluated after tanh and has bounded
    residuals because every action component lies in [-1, 1].  The clean
    branch is a target only; gradients update the shifted-image branch.
    """

    with torch.no_grad():
        target_action = bounded_action_mean(
            agent, rgb, proprio, augment=False,
        )
    shifted_action = bounded_action_mean(agent, rgb, proprio, augment=True)
    loss = F.smooth_l1_loss(shifted_action, target_action, beta=0.1)
    if not bool(torch.isfinite(loss)):
        raise ValueError("bounded shift-action consistency is non-finite")
    return loss
