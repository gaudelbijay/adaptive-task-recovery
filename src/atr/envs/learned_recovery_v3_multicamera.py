"""Training-only multicamera rendering of one LearnedRecovery-v3 physics state."""

from __future__ import annotations

import numpy as np

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from atr.envs.learned_recovery_v3 import LearnedRecoveryEventRewardEnv


CAMERA_EYES = {
    "base_camera": [0.45, 0.0, 0.72],
    "camera_left_5cm": [0.45, 0.05, 0.72],
    "camera_high_5cm": [0.45, 0.0, 0.77],
}


@register_env("LearnedRecovery-v3-MultiCamera", max_episode_steps=200)
class LearnedRecoveryMultiCameraEnv(LearnedRecoveryEventRewardEnv):
    """Expose three simultaneous RGB views without changing task physics."""

    @property
    def _default_sensor_configs(self):
        size = self.vision_camera_size
        target = np.asarray([0.05, 0.0, 0.04])
        return [
            CameraConfig(
                name,
                sapien_utils.look_at(eye=np.asarray(eye), target=target),
                size,
                size,
                np.pi / 2,
                0.01,
                100,
            )
            for name, eye in CAMERA_EYES.items()
        ]
