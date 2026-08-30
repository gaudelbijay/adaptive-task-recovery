#!/usr/bin/env python3
"""Isolated evaluator extension for the V28 unseen visual-domain suite."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import evaluate_visual_recovery_ppo as base
from evaluate_v28_visual_recovery import install_protocol_adapter
from atr.envs import learned_recovery_v3_ood as rendered


NEW_PROFILES = (
    "camera_right_3cm", "camera_low_3cm", "lighting_bright", "lighting_cool",
)
NEW_PERTURBATIONS = (
    "pixel_shift_left_3", "brightness_85", "cool_color_shift",
)
_base_camera_eye = rendered.camera_eye
_base_lighting_parameters = rendered.lighting_parameters
_base_perturbation = base.apply_visual_perturbation


def camera_eye(profile: str) -> np.ndarray:
    if profile not in NEW_PROFILES:
        return _base_camera_eye(profile)
    eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
    if profile == "camera_right_3cm":
        eye[1] -= 0.03
    elif profile == "camera_low_3cm":
        eye[2] -= 0.03
    return eye


def lighting_parameters(profile: str):
    if profile not in NEW_PROFILES:
        return _base_lighting_parameters(profile)
    if profile in ("camera_right_3cm", "camera_low_3cm"):
        return None
    if profile == "lighting_bright":
        return [0.45] * 3, [1.30] * 3, [0.85] * 3
    return [0.15, 0.22, 0.32], [0.45, 0.70, 1.05], [0.30, 0.48, 0.72]


def apply_visual_perturbation(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in NEW_PERTURBATIONS:
        return _base_perturbation(rgb, mode)
    if mode == "pixel_shift_left_3":
        image = rgb.permute(0, 3, 1, 2)
        height, width = image.shape[-2:]
        image = F.pad(image, (3, 3, 3, 3), mode="replicate")
        return image[:, :, 3:3 + height, 6:6 + width].permute(0, 2, 3, 1)
    image = rgb.float()
    if mode == "brightness_85":
        image = image * 0.85
    else:
        scale = torch.tensor(
            [0.80, 0.95, 1.15], device=image.device, dtype=image.dtype,
        )
        image = image * scale
    return image.round().clamp(0, 255).to(rgb.dtype)


def install_extensions() -> None:
    """Patch only this isolated evaluator process; shared source stays unchanged."""
    rendered.PROFILES = tuple(rendered.PROFILES) + NEW_PROFILES
    rendered.camera_eye = camera_eye
    rendered.lighting_parameters = lighting_parameters
    base.VISUAL_PERTURBATIONS = tuple(base.VISUAL_PERTURBATIONS) + NEW_PERTURBATIONS
    base.ENVIRONMENT_PROFILES = tuple(base.ENVIRONMENT_PROFILES) + NEW_PROFILES
    base.apply_visual_perturbation = apply_visual_perturbation


if __name__ == "__main__":
    install_extensions()
    install_protocol_adapter()
    base.main()
