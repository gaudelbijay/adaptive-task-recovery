"""DrAC-style policy invariance for tanh-transformed Gaussian actors.

The PPO importance ratio must be computed on the original observation.  This
module supplies a separate consistency loss between the frozen policy on that
observation and the live policy on a random-shift augmentation.  Because tanh
is a bijection from the real line to the open action interval, KL is invariant
under that shared transform and can be computed exactly on the pre-tanh
diagonal Normals.
"""

from __future__ import annotations

import torch


def diagonal_gaussian_kl(
    target_mean: torch.Tensor,
    target_logstd: torch.Tensor,
    augmented_mean: torch.Tensor,
    augmented_logstd: torch.Tensor,
) -> torch.Tensor:
    """Mean KL(target || augmented), detaching the target distribution."""

    if not (
        target_mean.shape == target_logstd.shape
        == augmented_mean.shape == augmented_logstd.shape
    ):
        raise ValueError("DrAC Gaussian parameters must have identical shapes")
    target_mean = target_mean.detach()
    target_logstd = target_logstd.detach()
    target_variance = (2 * target_logstd).exp()
    augmented_variance = (2 * augmented_logstd).exp()
    per_dimension = (
        augmented_logstd - target_logstd
        + (target_variance + (target_mean - augmented_mean).square())
        / (2 * augmented_variance)
        - 0.5
    )
    loss = per_dimension.sum(dim=-1).mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("DrAC policy KL is non-finite")
    return loss


def policy_parameters(agent, rgb, proprio, *, augment: bool):
    latent = agent.encode(rgb, augment=augment)
    actor_parts = [latent, proprio]
    if agent.goal_progress_predictor is not None:
        actor_parts.append(torch.sigmoid(agent.goal_progress_predictor(latent)))
    mean = agent.actor(torch.cat(actor_parts, dim=1))
    logstd = agent.actor_logstd.expand_as(mean)
    return mean, logstd


def drac_policy_consistency(agent, rgb, proprio) -> torch.Tensor:
    """Compare original-observation target policy with augmented live policy."""

    with torch.no_grad():
        target_mean, target_logstd = policy_parameters(
            agent, rgb, proprio, augment=False,
        )
    augmented_mean, augmented_logstd = policy_parameters(
        agent, rgb, proprio, augment=True,
    )
    return diagonal_gaussian_kl(
        target_mean, target_logstd, augmented_mean, augmented_logstd,
    )
