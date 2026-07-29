"""Object-level intervention spike — the actual gating test for I-003/D-006.

Standing balance (humanoid_stand_spike.py) turned out not to be the hard
question. What actually matters for the research question is whether the
simulator can realize the kinds of `WorldIntervention` docs/04-benchmark-environment.md
describes: an object removed/destroyed, a route blocked, an object locked
away. This tests that directly, on a small tabletop scene (embodiment
doesn't matter here, so this uses ManiSkill3's plain `panda` arm, not a
humanoid) with two interventions:

- `intervention_kind="object_removed"`: physically removes an actor from the
  scene mid-episode (`actor.remove_from_scene()`, CPU-sim only) — tests
  "required object removed/destroyed."
- `intervention_kind="route_blocked"`: spawns a brand new actor into the
  scene mid-episode — tests something removal doesn't: can new geometry be
  added after the scene is already built, not just mutated.

Both record ground-truth before/after state, matching the before/after +
oracle-effect logging the Intervention API sketch calls for.

CPU sim only, by simulator design, not by choice: GPU-batched sim
pre-allocates fixed per-actor buffers at reconfigure time, so it has no
supported path for removing or adding actors mid-episode. `_trigger_intervention`
raises a clear `RuntimeError` if this is instantiated under GPU sim rather
than failing silently or cryptically. Always pass
`sim_backend="physx_cpu"` (or `resolve_sim_backend(prefer_gpu=False)`) for
this specific env, regardless of what's available on the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import build_box
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@dataclass
class InterventionRecord:
    kind: Literal["object_removed", "route_blocked"]
    onset_step: int
    target_name: str
    pose_before: np.ndarray
    exists_after: bool
    pose_after: np.ndarray | None


class ObjectInterventionSpikeEnv(BaseEnv):
    """Tabletop scene: one 'target' object + one persistent 'landmark' object,
    plus one scripted object-level intervention triggered mid-episode."""

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        intervention_kind: Literal["object_removed", "route_blocked"] = "object_removed",
        onset_step_range: tuple[int, int] = (5, 15),
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._record: InterventionRecord | None = None
        self._target = None
        self._landmark = None
        self._blocker = None
        self._target_exists = True
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.6, 0.8], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        build_ground(self.scene)
        self._target = build_box(
            self.scene, half_sizes=[0.03, 0.03, 0.03], color=[1, 0, 0, 1],
            name="target_object", body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.3, 0.0, 0.03]),
        )
        self._landmark = build_box(
            self.scene, half_sizes=[0.03, 0.03, 0.03], color=[0, 1, 0, 1],
            name="landmark_object", body_type="dynamic",
            initial_pose=sapien.Pose(p=[0.3, 0.2, 0.03]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        seed = int(self._episode_rng.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        self._onset_step = int(rng.integers(*self.onset_step_range))
        self._elapsed_control_steps = 0
        self._record = None
        self._blocker = None
        self._target_exists = True

    def _before_control_step(self):
        if self._record is None and self._elapsed_control_steps == self._onset_step:
            self._trigger_intervention()
        self._elapsed_control_steps += 1

    def _trigger_intervention(self):
        if self.scene.gpu_sim_enabled:
            # Both interventions mutate the scene's actor set at runtime
            # (removing one / adding one). GPU sim pre-allocates fixed-size
            # per-actor buffers at reconfigure time across all parallel
            # sub-scenes, so this isn't a "we didn't get to it yet" gap —
            # SAPIEN's own Actor.remove_from_scene() explicitly documents
            # CPU-only support, and there's no GPU-sim equivalent for adding
            # a brand new actor mid-episode either. Fail loudly and early
            # rather than silently corrupting sim state or hitting a cryptic
            # low-level error several calls deep.
            raise RuntimeError(
                f"{self.intervention_kind} intervention requires CPU sim "
                "(scene.gpu_sim_enabled=True) — object add/remove is a "
                "structural scene mutation GPU-batched sim doesn't support. "
                "Use resolve_sim_backend(prefer_gpu=False) or pass "
                "sim_backend='physx_cpu' for this env."
            )
        pose_before = self._target.pose.sp.p.copy()
        if self.intervention_kind == "object_removed":
            self._target.remove_from_scene()
            # Finding: `remove_from_scene()` genuinely removes the entity from
            # the low-level SAPIEN physics scene (verified via
            # scene.sub_scenes[0].entities membership), but the high-level
            # `Actor` Python wrapper (`self._target`) goes stale afterward —
            # `.pose` and `.px_body_type` keep returning cached pre-removal
            # values instead of erroring or reflecting removal. Any
            # oracle/eval code must track existence itself; it cannot query
            # the wrapper post-removal.
            self._target_exists = False
            self._record = InterventionRecord(
                kind="object_removed", onset_step=self._onset_step,
                target_name="target_object", pose_before=pose_before,
                exists_after=False, pose_after=None,
            )
        elif self.intervention_kind == "route_blocked":
            self._blocker = build_box(
                self.scene, half_sizes=[0.05, 0.15, 0.1], color=[0.2, 0.2, 0.2, 1],
                name="route_blocker", body_type="static",
                initial_pose=sapien.Pose(p=[0.3, 0.1, 0.1]),
            )
            self.scene.update_render()
            self._record = InterventionRecord(
                kind="route_blocked", onset_step=self._onset_step,
                target_name="route_blocker", pose_before=pose_before,
                exists_after=True, pose_after=self._blocker.pose.sp.p.copy(),
            )

    def evaluate(self):
        return {
            "intervention_applied": self._record is not None,
            "target_exists": self._target_exists,
        }

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    @property
    def last_intervention_record(self) -> InterventionRecord | None:
        return self._record


@register_env("ObjectInterventionSpike-v1", max_episode_steps=50)
class ObjectInterventionSpikeRegisteredEnv(ObjectInterventionSpikeEnv):
    pass
