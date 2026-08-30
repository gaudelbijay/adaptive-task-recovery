"""Evaluation-only rendered visual-domain shifts for LearnedRecovery-v3."""

from __future__ import annotations

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from atr.envs.learned_recovery_v3 import LearnedRecoveryEventRewardEnv


PROFILES = (
    "camera_left_5cm", "camera_high_5cm", "lighting_dim", "lighting_warm",
)


def camera_eye(profile: str) -> np.ndarray:
    if profile not in PROFILES:
        raise ValueError(f"unknown visual-domain profile: {profile}")
    eye = np.asarray([0.45, 0.0, 0.72], dtype=float)
    if profile == "camera_left_5cm":
        eye[1] += 0.05
    elif profile == "camera_high_5cm":
        eye[2] += 0.05
    return eye


def lighting_parameters(profile: str) -> tuple[list[float], list[float], list[float]] | None:
    if profile not in PROFILES:
        raise ValueError(f"unknown visual-domain profile: {profile}")
    if profile in ("camera_left_5cm", "camera_high_5cm"):
        return None
    if profile == "lighting_dim":
        return [0.12] * 3, [0.45] * 3, [0.35] * 3
    return [0.30, 0.22, 0.15], [1.0, 0.72, 0.42], [0.65, 0.45, 0.28]


@register_env("LearnedRecovery-v3-OOD", max_episode_steps=200)
class LearnedRecoveryVisualOODEnv(LearnedRecoveryEventRewardEnv):
    """Keep task physics fixed while shifting camera pose or renderer lights."""

    def __init__(self, *args, visual_domain_profile: str, **kwargs):
        if visual_domain_profile not in PROFILES:
            raise ValueError(f"unknown visual-domain profile: {visual_domain_profile}")
        self.visual_domain_profile = visual_domain_profile
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        eye = camera_eye(self.visual_domain_profile)
        pose = sapien_utils.look_at(eye=eye, target=[0.05, 0.0, 0.04])
        size = self.vision_camera_size
        return [CameraConfig("base_camera", pose, size, size, np.pi / 2, 0.01, 100)]

    def _load_lighting(self, options: dict):
        parameters = lighting_parameters(self.visual_domain_profile)
        if parameters is None:
            return super()._load_lighting(options)
        shadow = self.enable_shadow
        ambient, key, fill = parameters
        self.scene.set_ambient_light(ambient)
        self.scene.add_directional_light(
            [1, 1, -1], key, shadow=shadow,
            shadow_scale=5, shadow_map_size=2048,
        )
        self.scene.add_directional_light([0, 0, -1], fill)
