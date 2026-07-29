"""HumanoidStandSpike — ManiSkill3 simulator-selection spike, not project architecture.

Per D-006 in ai-notes/decisions.md ("No simulator-specific architecture
should be committed before it passes the selection criteria"), this is
investigation code, not a benchmark environment. It tests exactly one thing
against docs/04-benchmark-environment.md's "Selection requirements": does
ManiSkill3 load a humanoid asset, run stably enough to be usable, and support
a deterministic seeded scripted event mid-episode? See ../README.md for
findings.

Built by extending ManiSkill3's own `UnitreeG1Stand`/`UnitreeH1Stand` pattern
(mani_skill/envs/tasks/humanoid/humanoid_stand.py) with one addition: a
scripted, seeded push via `ScriptedPushIntervention`.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import UnitreeG1Simplified, UnitreeH1Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from maniskill_humanoid_spike.scripted_intervention import (
    PushInterventionEvent,
    ScriptedPushIntervention,
)

_TORSO_LINK_NAMES = ("torso_link", "pelvis_contour_link", "pelvis")


class HumanoidStandSpikeEnv(BaseEnv):
    """Base class: standing task + one scripted push per episode.

    Concrete robot embodiments are registered as separate gym env ids below,
    matching the pattern ManiSkill3 itself uses for `UnitreeG1Stand-v1` /
    `UnitreeH1Stand-v1`, since different embodiments need different init
    poses / camera configs to run and render well.
    """

    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(
        self,
        *args,
        robot_uids: str,
        robot_init_qpos_noise: float = 0.02,
        push_onset_step_range: tuple[int, int] = (20, 80),
        push_force_range: tuple[float, float] = (80.0, 400.0),
        push_duration_steps: int = 5,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.push_onset_step_range = push_onset_step_range
        self.push_force_range = push_force_range
        self.push_duration_steps = push_duration_steps
        self._intervention: ScriptedPushIntervention | None = None
        self._last_intervention_event: PushInterventionEvent | None = None
        self._active_force: np.ndarray | None = None
        self._active_force_remaining = 0
        self._elapsed_control_steps = 0
        self._push_link = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            )
        )

    def _load_scene(self, options: dict):
        build_ground(self.scene)

    def _get_push_link(self):
        if self._push_link is None:
            links_by_name = {link.name: link for link in self.agent.robot.links}
            for name in _TORSO_LINK_NAMES:
                if name in links_by_name:
                    self._push_link = links_by_name[name]
                    break
            if self._push_link is None:
                self._push_link = self.agent.robot.links[0]
        return self._push_link

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            standing_keyframe = self.agent.keyframes["standing"]
            random_qpos = (
                torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float)
                * self.robot_init_qpos_noise
            )
            random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
            self.agent.robot.set_qpos(random_qpos)
            self.agent.robot.set_pose(sapien.Pose(p=standing_keyframe.pose.p))

        seed = int(self._episode_rng.randint(0, 2**31 - 1))
        self._intervention = ScriptedPushIntervention(
            np.random.default_rng(seed),
            onset_step_range=self.push_onset_step_range,
            force_magnitude_range=self.push_force_range,
        )
        self._last_intervention_event = None
        self._active_force = None
        self._active_force_remaining = 0
        self._elapsed_control_steps = 0

    def _before_control_step(self):
        event = self._intervention.maybe_trigger(self._elapsed_control_steps)
        if event is not None:
            self._active_force = event.force
            self._active_force_remaining = self.push_duration_steps
            self._last_intervention_event = event
        self._elapsed_control_steps += 1

    def _before_simulation_step(self):
        if self._active_force_remaining > 0:
            link = self._get_push_link()
            if self.scene.gpu_sim_enabled:
                # CPU sim exposes add_force_at_point() per-body; GPU sim is
                # batched and needs a direct write into the shared CUDA force
                # buffer instead (same pattern mani_skill.utils.structs.actor
                # .Actor.apply_force uses for its GPU branch).
                force = torch.as_tensor(self._active_force, dtype=torch.float32, device=self.device)
                self.scene.px.cuda_rigid_body_force.torch()[link._body_data_index, :3] = force
                self.scene.px.gpu_apply_rigid_dynamic_force()
            else:
                body = link._bodies[0]
                body.add_force_at_point(force=self._active_force, point=body.pose.p)
            self._active_force_remaining -= 1

    def evaluate(self):
        is_standing = self.agent.is_standing()
        return {
            "is_standing": is_standing,
            "fail": ~is_standing,
            "push_applied": self._last_intervention_event is not None,
        }

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["is_standing"]

    @property
    def last_intervention_event(self) -> PushInterventionEvent | None:
        """Ground-truth label for the push this episode — spike-eval use only."""
        return self._last_intervention_event


@register_env("HumanoidStandSpike-G1-v1", max_episode_steps=200)
class HumanoidStandSpikeG1Env(HumanoidStandSpikeEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_legs"]
    agent: Union[UnitreeG1Simplified]

    def __init__(self, *args, robot_uids="unitree_g1_simplified_legs", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)


@register_env("HumanoidStandSpike-H1-v1", max_episode_steps=200)
class HumanoidStandSpikeH1Env(HumanoidStandSpikeEnv):
    SUPPORTED_ROBOTS = ["unitree_h1_simplified"]
    agent: Union[UnitreeH1Simplified]

    def __init__(self, *args, robot_uids="unitree_h1_simplified", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
