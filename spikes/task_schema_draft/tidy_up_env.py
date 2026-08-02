"""TidyUp — a runnable draft of the docs/04-benchmark-environment.md task
schema, built around the project's own canonical example (docs/01
"Example"): "Put the red mug and blue bowl on the tray, keep the medicine
upright, and do not move the glass."

DRAFT for the "Shared: select the task family and irreversible/reversible
intervention set" item in STATUS.md — not a committed benchmark environment.
See ../README.md.

Scene: five objects on a tabletop (red_mug, blue_bowl, tray, medicine_bottle,
glass) plus an idle `panda` arm (no manipulation is exercised here — this
tests the world-state/intervention/oracle wiring, not skill execution).

Two scripted interventions, matched per docs/04's "Include matched
reversible and temporary changes. Otherwise the model may learn that every
detected change implies abandonment":

- `bowl_destroyed` (irreversible): removes blue_bowl mid-episode — the
  place_blue_bowl goal becomes infeasible; place_red_mug does not. (Goal ids
  come from instruction_parser.py's parser now, not hand-authored — see goal_graph.py's
  canonical_example() docstring.)
- `temporary_obstacle` (reversible/matched control): spawns a distractor
  object near the tray, then removes it again a few steps later — a
  detectable world change that never makes any goal infeasible.

CPU sim only, for the same reason as object_intervention_spike.py: object
add/remove is unsupported under GPU-batched sim.
"""

from __future__ import annotations

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

from task_schema_draft.goal_graph import CANONICAL_INSTRUCTION_TEXT, CANONICAL_OBJECTS
from task_schema_draft.instruction_parser import parse_instruction
from task_schema_draft.oracle_feasibility import ObjectState, WorldState, evaluate_goal_graph

_OBJECT_SPECS = {
    # name: (half_sizes, color, initial_pos)
    "red_mug": ([0.03, 0.03, 0.04], [0.8, 0.1, 0.1, 1], [0.15, -0.15, 0.04]),
    "blue_bowl": ([0.04, 0.04, 0.025], [0.1, 0.2, 0.8, 1], [0.15, 0.15, 0.025]),
    "tray": ([0.15, 0.2, 0.005], [0.5, 0.5, 0.5, 1], [0.4, 0.0, 0.005]),
    "medicine_bottle": ([0.015, 0.015, 0.05], [0.9, 0.9, 0.2, 1], [0.0, -0.2, 0.05]),
    "glass": ([0.02, 0.02, 0.045], [0.7, 0.9, 1.0, 0.6], [0.0, 0.2, 0.045]),
}


class TidyUpEnv(BaseEnv):
    """Draft task-schema environment — see module docstring."""

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        intervention_kind: Literal["bowl_destroyed", "temporary_obstacle", "none"] = "bowl_destroyed",
        onset_step_range: tuple[int, int] = (5, 15),
        obstacle_duration_steps: int = 10,
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self.obstacle_duration_steps = obstacle_duration_steps
        self.goal_graph = parse_instruction(CANONICAL_INSTRUCTION_TEXT, CANONICAL_OBJECTS)
        self._objects: dict[str, Any] = {}
        self._exists: dict[str, bool] = {}
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step: int | None = None
        self._initial_state: WorldState | None = None
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
        pose = sapien_utils.look_at([0.6, 0.6, 0.9], [0.15, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        build_ground(self.scene)
        self._objects = {}
        for name, (half_sizes, color, pos) in _OBJECT_SPECS.items():
            self._objects[name] = build_box(
                self.scene, half_sizes=half_sizes, color=color, name=name,
                body_type="dynamic", initial_pose=sapien.Pose(p=pos),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "TidyUpEnv requires CPU sim — object add/remove is unsupported "
                "under GPU-batched sim. Pass sim_backend='physx_cpu'."
            )
        seed = int(self._episode_rng.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        self._onset_step = int(rng.integers(*self.onset_step_range))
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step = None
        self._exists = {name: True for name in _OBJECT_SPECS}
        self._initial_state = None  # captured on the first evaluate() call after reset

    def _before_control_step(self):
        if (
            self.intervention_kind != "none"
            and not self._triggered
            and self._elapsed_control_steps == self._onset_step
        ):
            self._trigger_intervention()
        if (
            self._obstacle_remove_step is not None
            and self._elapsed_control_steps == self._obstacle_remove_step
        ):
            self._obstacle.remove_from_scene()
            self._obstacle = None
        self._elapsed_control_steps += 1

    def _trigger_intervention(self):
        self._triggered = True
        if self.intervention_kind == "bowl_destroyed":
            self._objects["blue_bowl"].remove_from_scene()
            self._exists["blue_bowl"] = False
        elif self.intervention_kind == "temporary_obstacle":
            self._obstacle = build_box(
                self.scene, half_sizes=[0.03, 0.03, 0.08], color=[0.3, 0.3, 0.3, 1],
                name="distractor_obstacle", body_type="static",
                initial_pose=sapien.Pose(p=[0.4, -0.25, 0.08]),  # near, not on, the tray
            )
            self.scene.update_render()
            self._obstacle_remove_step = self._elapsed_control_steps + self.obstacle_duration_steps

    def _world_state(self) -> WorldState:
        state: WorldState = {}
        for name in _OBJECT_SPECS:
            if not self._exists[name]:
                state[name] = ObjectState(exists=False, position=None, up_vector=None)
                continue
            obj = self._objects[name]
            pose = obj.pose.sp
            up_vector = pose.to_transformation_matrix()[:3, 2]
            state[name] = ObjectState(exists=True, position=pose.p.copy(), up_vector=up_vector.copy())
        return state

    def evaluate(self):
        current_state = self._world_state()
        if self._initial_state is None:
            self._initial_state = current_state
        oracle = evaluate_goal_graph(self.goal_graph, self._initial_state, current_state)
        return {
            "goal_feasibility": oracle["goal_feasibility"],
            "constraint_violations": oracle["constraint_violations"],
            "intervention_triggered": self._triggered,
            "obstacle_present": self._obstacle is not None,
        }

    def _get_obs_extra(self, info: dict):
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)


@register_env("TidyUpTaskSchemaDraft-v1", max_episode_steps=50)
class TidyUpRegisteredEnv(TidyUpEnv):
    pass
