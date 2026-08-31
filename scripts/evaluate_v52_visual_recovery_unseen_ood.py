#!/usr/bin/env python3
"""Isolated V52 evaluator for the frozen seed-127M renderer suite."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

import evaluate_visual_recovery_ppo as base
from atr.envs import learned_recovery_v3_ood as rendered
from evaluate_v44_visual_recovery import install_protocol_adapter
from v52_subpixel_specialist_agent import SubpixelSpecialistAgent


PROFILES = (
    "camera_left_front_5cm", "camera_yaw_left_3deg",
    "lighting_magenta_ambient", "lighting_low_side",
)
PERTURBATIONS = (
    "subpixel_shift_left_3_5", "rotation_clockwise_6deg",
    "scale_90", "combined_similarity_v2",
)
_camera_eye = rendered.camera_eye
_lighting_parameters = rendered.lighting_parameters
_sensor_configs = rendered.LearnedRecoveryVisualOODEnv._default_sensor_configs
_load_lighting = rendered.LearnedRecoveryVisualOODEnv._load_lighting
_perturbation = base.apply_visual_perturbation


def camera_eye(profile):
    if profile not in PROFILES:
        return _camera_eye(profile)
    target = np.asarray([0.05, 0.0, 0.04], dtype=float)
    eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
    if profile == "camera_left_front_5cm":
        direction = eye - target
        eye -= 0.05 * direction / np.linalg.norm(direction)
        eye[1] += 0.05
    elif profile == "camera_yaw_left_3deg":
        angle = math.radians(3.0)
        offset = eye - target
        eye = target + np.asarray([
            math.cos(angle) * offset[0] - math.sin(angle) * offset[1],
            math.sin(angle) * offset[0] + math.cos(angle) * offset[1],
            offset[2],
        ])
    return eye


def lighting_parameters(profile):
    if profile not in PROFILES:
        return _lighting_parameters(profile)
    if profile.startswith("camera_"):
        return None
    if profile == "lighting_magenta_ambient":
        return [0.32, 0.10, 0.32], [0.75] * 3, [0.28] * 3
    return [0.08] * 3, [0.70] * 3, [0.18] * 3


def sensor_configs(environment):
    if environment.visual_domain_profile not in ("camera_left_front_5cm", "camera_yaw_left_3deg"):
        return _sensor_configs.fget(environment)
    pose = sapien_utils.look_at(eye=camera_eye(environment.visual_domain_profile), target=[0.05, 0.0, 0.04])
    size = environment.vision_camera_size
    return [CameraConfig("base_camera", pose, size, size, np.pi / 2, 0.01, 100)]


def load_lighting(environment, options):
    if environment.visual_domain_profile not in ("lighting_magenta_ambient", "lighting_low_side"):
        return _load_lighting(environment, options)
    ambient, key, fill = lighting_parameters(environment.visual_domain_profile)
    environment.scene.set_ambient_light(ambient)
    direction = [1, -1, -1] if environment.visual_domain_profile == "lighting_magenta_ambient" else [-1, 0.3, -0.25]
    environment.scene.add_directional_light(direction, key, shadow=environment.enable_shadow,
                                            shadow_scale=5, shadow_map_size=2048)
    environment.scene.add_directional_light([0, 0, -1], fill)


def affine(rgb, shift_right, angle_degrees, scale):
    image = rgb.permute(0, 3, 1, 2).float(); angle = math.radians(angle_degrees); inverse = 1.0 / scale
    theta = torch.tensor([[[math.cos(angle)*inverse, math.sin(angle)*inverse, -2.0*shift_right/(rgb.shape[2]-1)],
                           [-math.sin(angle)*inverse, math.cos(angle)*inverse, 0.0]]], device=image.device, dtype=image.dtype)
    grid = F.affine_grid(theta.expand(image.shape[0], -1, -1), image.shape, align_corners=True)
    output = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return output.permute(0, 2, 3, 1).round().clamp(0, 255).to(rgb.dtype)


def apply_visual_perturbation(rgb, mode):
    if mode not in PERTURBATIONS:
        return _perturbation(rgb, mode)
    if mode == "subpixel_shift_left_3_5": return affine(rgb, -3.5, 0.0, 1.0)
    if mode == "rotation_clockwise_6deg": return affine(rgb, 0.0, -6.0, 1.0)
    if mode == "scale_90": return affine(rgb, 0.0, 0.0, 0.90)
    return affine(rgb, -3.5, -6.0, 0.90)


def install_extensions():
    rendered.PROFILES = tuple(rendered.PROFILES) + PROFILES
    rendered.camera_eye = camera_eye; rendered.lighting_parameters = lighting_parameters
    rendered.LearnedRecoveryVisualOODEnv._default_sensor_configs = property(sensor_configs)
    rendered.LearnedRecoveryVisualOODEnv._load_lighting = load_lighting
    base.VISUAL_PERTURBATIONS = tuple(base.VISUAL_PERTURBATIONS) + PERTURBATIONS
    base.ENVIRONMENT_PROFILES = tuple(base.ENVIRONMENT_PROFILES) + PROFILES
    base.apply_visual_perturbation = apply_visual_perturbation


if __name__ == "__main__":
    install_extensions(); install_protocol_adapter("subpixel_specialist_router_v19")
    base.VisualAgent = SubpixelSpecialistAgent
    base.main()
