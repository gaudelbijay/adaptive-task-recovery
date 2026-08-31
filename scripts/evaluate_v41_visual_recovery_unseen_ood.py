#!/usr/bin/env python3
"""Isolated evaluator for the frozen V41 untouched visual-domain suite."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import evaluate_visual_recovery_ppo as base
from evaluate_v41_visual_recovery import install_protocol_adapter
from atr.envs import learned_recovery_v3_ood as rendered


NEW_PROFILES = (
    "camera_right_back_4cm", "lighting_bright_side", "lighting_green_ambient",
)
NEW_PERTURBATIONS = (
    "subpixel_shift_right_2_25", "rotation_counterclockwise_4deg",
    "scale_108", "combined_similarity_v1",
)
_base_camera_eye = rendered.camera_eye
_base_lighting_parameters = rendered.lighting_parameters
_base_sensor_configs = rendered.LearnedRecoveryVisualOODEnv._default_sensor_configs
_base_load_lighting = rendered.LearnedRecoveryVisualOODEnv._load_lighting
_base_perturbation = base.apply_visual_perturbation


def camera_eye(profile: str) -> np.ndarray:
    if profile not in NEW_PROFILES:
        return _base_camera_eye(profile)
    eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
    if profile == "camera_right_back_4cm":
        target = np.asarray([0.05, 0.0, 0.04], dtype=float)
        direction = eye - target
        eye += 0.04 * direction / np.linalg.norm(direction)
        eye[1] -= 0.04
    return eye


def lighting_parameters(profile: str):
    if profile not in NEW_PROFILES:
        return _base_lighting_parameters(profile)
    if profile == "camera_right_back_4cm":
        return None
    if profile == "lighting_green_ambient":
        return [0.10, 0.32, 0.12], [0.75] * 3, [0.30] * 3
    # A brighter key from the opposite side is installed by load_lighting.
    return [0.24] * 3, [1.20] * 3, [0.45] * 3


def sensor_configs(environment):
    if environment.visual_domain_profile != "camera_right_back_4cm":
        return _base_sensor_configs.fget(environment)
    pose = sapien_utils.look_at(
        eye=camera_eye(environment.visual_domain_profile), target=[0.05, 0.0, 0.04],
    )
    size = environment.vision_camera_size
    return [CameraConfig("base_camera", pose, size, size, np.pi / 2, 0.01, 100)]


def load_lighting(environment, options: dict):
    if environment.visual_domain_profile not in (
        "lighting_bright_side", "lighting_green_ambient",
    ):
        return _base_load_lighting(environment, options)
    ambient, key, fill = lighting_parameters(environment.visual_domain_profile)
    environment.scene.set_ambient_light(ambient)
    key_direction = [-1, 1, -1] if environment.visual_domain_profile == "lighting_bright_side" else [1, 1, -1]
    environment.scene.add_directional_light(
        key_direction, key, shadow=environment.enable_shadow,
        shadow_scale=5, shadow_map_size=2048,
    )
    environment.scene.add_directional_light([0, 0, -1], fill)


def _affine(rgb: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2).float()
    transform = theta.to(device=image.device, dtype=image.dtype).expand(image.shape[0], -1, -1)
    grid = F.affine_grid(transform, image.shape, align_corners=True)
    transformed = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return transformed.permute(0, 2, 3, 1).round().clamp(0, 255).to(rgb.dtype)


def _similarity_theta(rgb: torch.Tensor, *, shift_right: float, angle_degrees: float,
                      output_scale: float) -> torch.Tensor:
    # affine_grid maps output coordinates to input coordinates, hence the
    # inverse scale and negative x translation for a rightward output shift.
    angle = math.radians(angle_degrees)
    inverse_scale = 1.0 / output_scale
    cosine = math.cos(angle) * inverse_scale
    sine = math.sin(angle) * inverse_scale
    translation = -2.0 * shift_right / (rgb.shape[2] - 1)
    return torch.tensor([[[cosine, sine, translation], [-sine, cosine, 0.0]]])


def apply_visual_perturbation(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in NEW_PERTURBATIONS:
        return _base_perturbation(rgb, mode)
    if mode == "subpixel_shift_right_2_25":
        theta = _similarity_theta(rgb, shift_right=2.25, angle_degrees=0.0, output_scale=1.0)
    elif mode == "rotation_counterclockwise_4deg":
        theta = _similarity_theta(rgb, shift_right=0.0, angle_degrees=4.0, output_scale=1.0)
    elif mode == "scale_108":
        theta = _similarity_theta(rgb, shift_right=0.0, angle_degrees=0.0, output_scale=1.08)
    else:
        theta = _similarity_theta(rgb, shift_right=2.25, angle_degrees=4.0, output_scale=1.08)
    return _affine(rgb, theta)


def install_extensions() -> None:
    """Patch only this evaluator process; training/shared source stays unchanged."""

    rendered.PROFILES = tuple(rendered.PROFILES) + NEW_PROFILES
    rendered.camera_eye = camera_eye
    rendered.lighting_parameters = lighting_parameters
    rendered.LearnedRecoveryVisualOODEnv._default_sensor_configs = property(sensor_configs)
    rendered.LearnedRecoveryVisualOODEnv._load_lighting = load_lighting
    base.VISUAL_PERTURBATIONS = tuple(base.VISUAL_PERTURBATIONS) + NEW_PERTURBATIONS
    base.ENVIRONMENT_PROFILES = tuple(base.ENVIRONMENT_PROFILES) + NEW_PROFILES
    base.apply_visual_perturbation = apply_visual_perturbation


if __name__ == "__main__":
    install_extensions()
    install_protocol_adapter()
    base.main()
