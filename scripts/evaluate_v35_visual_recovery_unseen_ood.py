#!/usr/bin/env python3
"""Isolated evaluator for the untouched D-176 V35 confirmation suite."""

from __future__ import annotations

import math

import numpy as np
import sapien
import torch
import torch.nn.functional as F
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import evaluate_visual_recovery_ppo as base
from evaluate_v35_visual_recovery import install_protocol_adapter
from atr.envs import learned_recovery_v3_ood as rendered


NEW_PROFILES = (
    "camera_back_3cm", "camera_roll_right_2deg", "lighting_cool",
    "lighting_back_key",
)
NEW_PERTURBATIONS = (
    "subpixel_shift_left_1_5", "rotation_clockwise_2deg", "scale_95",
)
_base_camera_eye = rendered.camera_eye
_base_lighting_parameters = rendered.lighting_parameters
_base_sensor_configs = rendered.LearnedRecoveryVisualOODEnv._default_sensor_configs
_base_load_lighting = rendered.LearnedRecoveryVisualOODEnv._load_lighting
_base_perturbation = base.apply_visual_perturbation


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply scalar-first quaternions without adding a new dependency."""

    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.size != 4 or right.size != 4:
        raise ValueError("quaternion inputs must each contain exactly four scalars")
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ])


def _flat_float32(value, size: int, description: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != size:
        raise ValueError(f"{description} must contain exactly {size} scalars")
    return result


def camera_eye(profile: str) -> np.ndarray:
    if profile not in NEW_PROFILES:
        return _base_camera_eye(profile)
    eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
    if profile == "camera_back_3cm":
        target = np.asarray([0.05, 0.0, 0.04], dtype=float)
        direction = eye - target
        eye += 0.03 * direction / np.linalg.norm(direction)
    return eye


def lighting_parameters(profile: str):
    if profile not in NEW_PROFILES:
        return _base_lighting_parameters(profile)
    if profile in ("camera_back_3cm", "camera_roll_right_2deg"):
        return None
    if profile == "lighting_cool":
        return [0.15, 0.22, 0.32], [0.45, 0.70, 1.05], [0.30, 0.48, 0.72]
    return [0.16] * 3, [0.85] * 3, [0.30] * 3


def sensor_configs(environment):
    if environment.visual_domain_profile != "camera_roll_right_2deg":
        return _base_sensor_configs.fget(environment)
    eye = camera_eye(environment.visual_domain_profile)
    pose = sapien_utils.look_at(eye=eye, target=[0.05, 0.0, 0.04])
    angle = math.radians(2.0)
    local_roll = np.asarray([math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0])
    position = _flat_float32(pose.p, 3, "camera position")
    orientation = _flat_float32(
        _quaternion_multiply(pose.q, local_roll), 4, "camera orientation",
    )
    rolled = sapien.Pose(p=position, q=orientation)
    size = environment.vision_camera_size
    return [CameraConfig("base_camera", rolled, size, size, np.pi / 2, 0.01, 100)]


def load_lighting(environment, options: dict):
    if environment.visual_domain_profile != "lighting_back_key":
        return _base_load_lighting(environment, options)
    environment.scene.set_ambient_light([0.16] * 3)
    environment.scene.add_directional_light(
        [-1, -1, -1], [0.85] * 3, shadow=environment.enable_shadow,
        shadow_scale=5, shadow_map_size=2048,
    )
    environment.scene.add_directional_light([0, 0, -1], [0.30] * 3)


def _affine(rgb: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    image = rgb.permute(0, 3, 1, 2).float()
    transform = theta.to(device=image.device, dtype=image.dtype).expand(image.shape[0], -1, -1)
    grid = F.affine_grid(transform, image.shape, align_corners=True)
    shifted = F.grid_sample(
        image, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return shifted.permute(0, 2, 3, 1).round().clamp(0, 255).to(rgb.dtype)


def apply_visual_perturbation(rgb: torch.Tensor, mode: str) -> torch.Tensor:
    if mode not in NEW_PERTURBATIONS:
        return _base_perturbation(rgb, mode)
    if mode == "subpixel_shift_left_1_5":
        translation = 2.0 * 1.5 / (rgb.shape[2] - 1)
        theta = torch.tensor([[[1.0, 0.0, translation], [0.0, 1.0, 0.0]]])
    elif mode == "rotation_clockwise_2deg":
        angle = math.radians(2.0)
        theta = torch.tensor([[[math.cos(angle), -math.sin(angle), 0.0],
                               [math.sin(angle), math.cos(angle), 0.0]]])
    else:
        inverse_scale = 1.0 / 0.95
        theta = torch.tensor([[[inverse_scale, 0.0, 0.0],
                               [0.0, inverse_scale, 0.0]]])
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
