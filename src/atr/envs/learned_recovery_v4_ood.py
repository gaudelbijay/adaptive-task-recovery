"""Evaluation-only renderer shifts for the mechanism-diverse V4 benchmark."""

from __future__ import annotations

import numpy as np
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from atr.envs.learned_recovery_v3_ood import PROFILES, lighting_parameters
from atr.envs.learned_recovery_v4 import LearnedRecoveryMechanismDiverseEnv


@register_env("LearnedRecovery-v4-OOD", max_episode_steps=240)
class LearnedRecoveryMechanismDiverseOODEnv(LearnedRecoveryMechanismDiverseEnv):
    """Shift only camera or lights while preserving V4 task physics."""

    def __init__(self, *args, visual_domain_profile: str, **kwargs):
        if visual_domain_profile not in PROFILES:
            raise ValueError(f"unknown visual-domain profile: {visual_domain_profile}")
        self.visual_domain_profile = visual_domain_profile
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
        if self.visual_domain_profile == "camera_left_5cm":
            eye[1] += 0.05
        elif self.visual_domain_profile == "camera_high_5cm":
            eye[2] += 0.05
        pose = sapien_utils.look_at(eye=eye, target=[0.05, 0.0, 0.04])
        size = self.vision_camera_size
        return [CameraConfig("base_camera", pose, size, size, np.pi / 2, 0.01, 100)]

    def _load_lighting(self, options: dict):
        parameters = lighting_parameters(self.visual_domain_profile)
        if parameters is None:
            return super()._load_lighting(options)
        ambient, key, fill = parameters
        self.scene.set_ambient_light(ambient)
        self.scene.add_directional_light(
            [1, 1, -1], key, shadow=self.enable_shadow,
            shadow_scale=5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], fill)
