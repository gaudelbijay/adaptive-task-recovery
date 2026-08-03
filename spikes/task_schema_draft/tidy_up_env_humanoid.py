"""TidyUp on a Unitree G1 humanoid — same schema/goals/interventions as
tidy_up_env.py (panda arm), different embodiment. Demonstrates that
goal_graph.py / oracle_feasibility.py / intent_guard.py are genuinely
embodiment-agnostic, per docs/04's "feasibility and intent modules are
embodiment-agnostic, while a humanoid control layer provides ... reaching,
grasping" framing.

DRAFT for the "Shared: select the task family and irreversible/reversible
intervention set" item in STATUS.md — not committed. See ../README.md.

Robot: `unitree_g1_simplified_upper_body_with_head_camera` — the same agent
class ManiSkill3's own `UnitreeG1PlaceAppleInBowl-v1` example uses. That
agent only exposes joint-space controllers (`pd_joint_pos`,
`pd_joint_delta_pos`) — no built-in Cartesian end-effector controller like
Panda's `pd_ee_delta_pos` (checked directly: the agent class just doesn't
define one). So instead of proportional IK toward an xyz target
(tidy_up_env.py's approach), this uses two hand-calibrated right-arm joint
configurations ("reach_mug", "reach_bowl") found empirically by sweeping
shoulder/elbow angles and reading off `agent.right_tcp.pose`. Placement
still uses the same teleport-on-success abstraction as the panda version,
for the same reason (the grasp mechanic itself isn't what's under test).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.unitree_g1.g1_upper_body import (
    UnitreeG1UpperBodyWithHeadCamera,
)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import build_box
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from atr.language.goal_graph import canonical_example
from atr.feasibility.oracle import ObjectState, WorldState, evaluate_goal_graph

_COUNTER_Z = 0.75

_OBJECT_SPECS = {
    # name: (half_sizes, color, initial_pos) -- xy calibrated to the two
    # reachable zones below; z matches the kitchen counter top height.
    "red_mug": ([0.025, 0.025, 0.035], [0.8, 0.1, 0.1, 1], [0.0, 0.0, _COUNTER_Z + 0.035]),
    "blue_bowl": ([0.035, 0.035, 0.02], [0.1, 0.2, 0.8, 1], [0.0, -0.27, _COUNTER_Z + 0.02]),
    "tray": ([0.12, 0.15, 0.005], [0.5, 0.5, 0.5, 1], [0.0, -0.13, _COUNTER_Z + 0.005]),
    "medicine_bottle": ([0.012, 0.012, 0.04], [0.9, 0.9, 0.2, 1], [0.15, 0.15, _COUNTER_Z + 0.04]),
    # x=-0.15 fell off the counter's left edge in testing (KitchenCounterSceneBuilder's
    # footprint isn't symmetric around x=0) -- kept on the proven-good x=+0.15 side instead.
    "glass": ([0.018, 0.018, 0.04], [0.7, 0.9, 1.0, 0.6], [0.15, -0.15, _COUNTER_Z + 0.04]),
}

# Empirically calibrated right-arm joint targets (pd_joint_pos, absolute).
# See module docstring — found by sweeping shoulder/elbow angles on
# UnitreeG1PlaceAppleInBowl-v1 and reading off agent.right_tcp.pose.sp.p.
_REACH_CONFIGS = {
    "red_mug": {
        "right_shoulder_pitch_joint": 1.0, "right_shoulder_roll_joint": 0.4,
        "right_shoulder_yaw_joint": 1.0, "right_elbow_pitch_joint": 1.4,
    },
    "blue_bowl": {
        "right_shoulder_pitch_joint": 1.2, "right_shoulder_roll_joint": -0.6,
        "right_shoulder_yaw_joint": -0.8, "right_elbow_pitch_joint": 1.6,
    },
}
_NEUTRAL_QPOS = np.zeros(25, dtype=np.float32)


class TidyUpHumanoidEnv(BaseEnv):
    """Same task as TidyUpEnv, on a Unitree G1 upper body instead of panda."""

    SUPPORTED_REWARD_MODES = ["none"]
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera
    kitchen_scene_scale = 0.82

    def __init__(
        self,
        *args,
        intervention_kind: Literal["bowl_destroyed", "temporary_obstacle", "none"] = "bowl_destroyed",
        onset_step_range: tuple[int, int] = (4, 6),  # after the settle window (_SETTLE_STEPS)
        obstacle_duration_steps: int = 10,
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self.obstacle_duration_steps = obstacle_duration_steps
        self.goal_graph = canonical_example()
        self._objects: dict[str, Any] = {}
        self._exists: dict[str, bool] = {}
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step: int | None = None
        self._initial_state: WorldState | None = None
        super().__init__(
            *args, robot_uids="unitree_g1_simplified_upper_body_with_head_camera", **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            scene_config=SceneConfig(contact_offset=0.01),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.9, -0.5, 1.3], [-0.1, -0.15, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)
        self._objects = {}
        for name, (half_sizes, color, pos) in _OBJECT_SPECS.items():
            self._objects[name] = build_box(
                self.scene, half_sizes=half_sizes, color=color, name=name,
                body_type="dynamic", initial_pose=sapien.Pose(p=pos),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "TidyUpHumanoidEnv requires CPU sim — object add/remove is "
                "unsupported under GPU-batched sim. Pass sim_backend='physx_cpu'."
            )
        self.scene_builder.initialize(env_idx)
        with torch.device(self.device):
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[-0.3, 0, 0.755]))

        seed = int(self._episode_rng.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        self._onset_step = int(rng.integers(*self.onset_step_range))
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step = None
        self._exists = {name: True for name in _OBJECT_SPECS}
        self._initial_state = None

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
                self.scene, half_sizes=[0.02, 0.02, 0.06], color=[0.3, 0.3, 0.3, 1],
                name="distractor_obstacle", body_type="static",
                initial_pose=sapien.Pose(p=[0.0, -0.13, _COUNTER_Z + 0.06]),
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

    _SETTLE_STEPS = 3

    def evaluate(self):
        current_state = self._world_state()
        # Objects are spawned at an assumed counter height that doesn't
        # exactly match the kitchen counter's real collision surface, so
        # they drop a small amount as they settle in the first few steps.
        # Keep refreshing the never_move/upright baseline through that
        # settle window so it freezes at the resting state, not the
        # pre-physics spawn pose (which would trip a false violation).
        if self._initial_state is None or self._elapsed_control_steps <= self._SETTLE_STEPS:
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


@register_env("TidyUpTaskSchemaDraft-Humanoid-v1", max_episode_steps=100)
class TidyUpHumanoidRegisteredEnv(TidyUpHumanoidEnv):
    pass
