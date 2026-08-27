"""GPU-batched, end-to-end learned manipulation recovery benchmark.

The Panda receives a continuous joint-space action on every control step.  It
must place a red and a blue cube on their matching goal pads in the order
specified by a symbolic language instruction.  Mid-episode, a dynamic sweeper
is accelerated with an external force and can physically knock one cube off
the table.  The correct response is to skip only the now-infeasible goal and
finish the feasible suffix while not moving the protected yellow object.

Pose assignment is confined to ``_initialize_episode``.  In particular, the
intervention uses ``Actor.apply_force`` and contact dynamics; neither task
execution nor recovery calls ``set_pose``.  This makes the environment useful
for learned-control comparisons without inheriting the project's older
navigate-then-teleport abstraction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("LearnedRecovery-v1", max_episode_steps=200)
class LearnedRecoveryEnv(BaseEnv):
    """Two-goal manipulation with a physical, irreversible intervention."""

    SUPPORTED_ROBOTS = ["panda"]
    cube_half_size = 0.025
    goal_threshold = 0.04

    def __init__(
        self,
        *args,
        robot_uids="panda",
        intervention_probability: float = 0.8,
        onset_step_range: tuple[int, int] = (18, 36),
        intervention_force: float = 2.0,
        intervention_steps: int = 12,
        oracle_observation: bool = False,
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        if not 0 <= intervention_probability <= 1:
            raise ValueError("intervention_probability must be in [0, 1]")
        self.intervention_probability = float(intervention_probability)
        self.onset_step_range = tuple(int(v) for v in onset_step_range)
        self.intervention_force = float(intervention_force)
        self.intervention_steps = int(intervention_steps)
        self.oracle_observation = bool(oracle_observation)
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        self._episode_step = None
        self._onset_step = None
        self._intervention_target = None
        self._completed = None
        self._constraint_violated = None
        self._protected_initial_position = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.45, 0.0, 0.72], target=[0.05, 0.0, 0.04])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.65, 0.65, 0.65], [0.05, 0.0, 0.05])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.red_cube = actors.build_cube(
            self.scene, self.cube_half_size, [0.9, 0.05, 0.05, 1], "red_cube",
            initial_pose=sapien.Pose(p=[0.0, -0.12, self.cube_half_size]),
        )
        self.blue_cube = actors.build_cube(
            self.scene, self.cube_half_size, [0.05, 0.15, 0.9, 1], "blue_cube",
            initial_pose=sapien.Pose(p=[0.0, 0.12, self.cube_half_size]),
        )
        self.red_goal = actors.build_box(
            self.scene, [0.045, 0.045, 0.002], [1.0, 0.25, 0.25, 1], "red_goal",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0.16, -0.25, 0.003]),
        )
        self.blue_goal = actors.build_box(
            self.scene, [0.045, 0.045, 0.002], [0.25, 0.35, 1.0, 1], "blue_goal",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0.16, 0.25, 0.003]),
        )
        self.red_sweeper = actors.build_box(
            self.scene, [0.025, 0.042, 0.025], [0.25, 0.25, 0.25, 1], "red_sweeper",
            initial_pose=sapien.Pose(p=[-0.16, -0.12, 0.025]),
        )
        self.blue_sweeper = actors.build_box(
            self.scene, [0.025, 0.042, 0.025], [0.25, 0.25, 0.25, 1], "blue_sweeper",
            initial_pose=sapien.Pose(p=[-0.16, 0.12, 0.025]),
        )
        self.protected = actors.build_box(
            self.scene, half_sizes=[0.022, 0.022, 0.055],
            color=[0.95, 0.8, 0.05, 1], name="protected_object",
            initial_pose=sapien.Pose(p=[0.08, 0.0, 0.055]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Randomized reset is the only method allowed to assign actor poses."""
        with torch.device(self.device):
            batch = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self._episode_step is None:
                self._episode_step = torch.zeros(self.num_envs, dtype=torch.long)
                self._onset_step = torch.zeros(self.num_envs, dtype=torch.long)
                self._intervention_target = torch.full(
                    (self.num_envs,), -1, dtype=torch.long
                )
                self._instruction_first = torch.zeros(self.num_envs, dtype=torch.long)
                self._completed = torch.zeros((self.num_envs, 2), dtype=torch.bool)
                self._constraint_violated = torch.zeros(self.num_envs, dtype=torch.bool)
                self._protected_initial_position = torch.zeros((self.num_envs, 3))

            self._episode_step[env_idx] = 0
            self._onset_step[env_idx] = torch.randint(
                self.onset_step_range[0], self.onset_step_range[1] + 1, (batch,)
            )
            has_intervention = torch.rand(batch) < self.intervention_probability
            target = torch.randint(0, 2, (batch,))
            self._intervention_target[env_idx] = torch.where(
                has_intervention, target, torch.full_like(target, -1)
            )
            self._instruction_first[env_idx] = torch.randint(0, 2, (batch,))
            self._completed[env_idx] = False
            self._constraint_violated[env_idx] = False

            jitter = (torch.rand((batch, 2)) - 0.5) * 0.035
            red_position = torch.zeros((batch, 3))
            red_position[:, 0] = jitter[:, 0]
            red_position[:, 1] = -0.12
            red_position[:, 2] = self.cube_half_size
            blue_position = torch.zeros((batch, 3))
            blue_position[:, 0] = jitter[:, 1]
            blue_position[:, 1] = 0.12
            blue_position[:, 2] = self.cube_half_size
            quaternion = randomization.random_quaternions(
                batch, lock_x=True, lock_y=True
            )
            self.red_cube.set_pose(Pose.create_from_pq(red_position, quaternion))
            quaternion = randomization.random_quaternions(
                batch, lock_x=True, lock_y=True
            )
            self.blue_cube.set_pose(Pose.create_from_pq(blue_position, quaternion))

            fixed = torch.tensor
            self.red_goal.set_pose(Pose.create_from_pq(fixed([[0.16, -0.25, 0.003]]).repeat(batch, 1)))
            self.blue_goal.set_pose(Pose.create_from_pq(fixed([[0.16, 0.25, 0.003]]).repeat(batch, 1)))
            self.red_sweeper.set_pose(Pose.create_from_pq(fixed([[-0.16, -0.12, 0.025]]).repeat(batch, 1)))
            self.blue_sweeper.set_pose(Pose.create_from_pq(fixed([[-0.16, 0.12, 0.025]]).repeat(batch, 1)))
            self.red_sweeper.set_linear_velocity(torch.zeros((batch, 3)))
            self.blue_sweeper.set_linear_velocity(torch.zeros((batch, 3)))
            protected_position = fixed([[0.08, 0.0, 0.055]]).repeat(batch, 1)
            self.protected.set_pose(Pose.create_from_pq(protected_position))
            self.protected.set_linear_velocity(torch.zeros((batch, 3)))
            self._protected_initial_position[env_idx] = protected_position

    def _before_simulation_step(self):
        # A force over multiple physics steps produces a contact-mediated
        # intervention. Zero rows ensure other vector environments are inert.
        active = (
            (self._episode_step >= self._onset_step)
            & (self._episode_step < self._onset_step + self.intervention_steps)
        )
        force = torch.zeros((self.num_envs, 3), device=self.device)
        force[:, 0] = self.intervention_force
        red_force = force * (active & (self._intervention_target == 0))[:, None]
        blue_force = force * (active & (self._intervention_target == 1))[:, None]
        # ManiSkill 3.0.0b22's CPU Actor wrapper forwards directly to SAPIEN,
        # whose scalar API requires a float32 (3,) NumPy array. GPU simulation
        # uses the batched tensor API that large-scale training needs.
        if self.scene.gpu_sim_enabled:
            self.red_sweeper.apply_force(red_force)
            self.blue_sweeper.apply_force(blue_force)
        else:
            self.red_sweeper.apply_force(red_force[0].detach().cpu().numpy().astype(np.float32))
            self.blue_sweeper.apply_force(blue_force[0].detach().cpu().numpy().astype(np.float32))

    def _after_control_step(self):
        self._episode_step += 1

    def _unavailable(self) -> torch.Tensor:
        positions = torch.stack([self.red_cube.pose.p, self.blue_cube.pose.p], dim=1)
        return (positions[:, :, 0] > 0.36) | (positions[:, :, 2] < -0.02)

    def _placed(self) -> torch.Tensor:
        cubes = torch.stack([self.red_cube.pose.p, self.blue_cube.pose.p], dim=1)
        goals = torch.stack([self.red_goal.pose.p, self.blue_goal.pose.p], dim=1)
        xy_close = torch.linalg.norm(cubes[:, :, :2] - goals[:, :, :2], dim=2) < self.goal_threshold
        on_table = (cubes[:, :, 2] > 0.015) & (cubes[:, :, 2] < 0.06)
        return xy_close & on_table

    def _update_task_memory(self):
        unavailable = self._unavailable()
        placed = self._placed()
        first = self._instruction_first
        second = 1 - first
        rows = torch.arange(self.num_envs, device=self.device)
        first_done = self._completed[rows, first] | unavailable[rows, first]
        accept_first = placed[rows, first]
        self._completed[rows, first] |= accept_first
        first_done |= accept_first
        self._completed[rows, second] |= placed[rows, second] & first_done
        displacement = torch.linalg.norm(
            self.protected.pose.p - self._protected_initial_position, dim=1
        )
        self._constraint_violated |= displacement > 0.015
        return unavailable

    def evaluate(self):
        unavailable = self._update_task_memory()
        resolved = self._completed | unavailable
        intervention_finished = self._episode_step >= (
            self._onset_step + self.intervention_steps
        )
        success = resolved.all(dim=1) & intervention_finished & ~self._constraint_violated
        return {
            "success": success,
            "goals_completed": self._completed.float().sum(dim=1),
            "goals_unavailable": unavailable.float().sum(dim=1),
            "constraint_violated": self._constraint_violated,
            "intervention_occurred": self._intervention_target >= 0,
        }

    def _instruction_encoding(self):
        # [red-first, blue-first] is the deterministic parse of the two
        # benchmark instruction templates, supplied to both state and vision.
        return torch.nn.functional.one_hot(self._instruction_first, 2).float()

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "instruction": self._instruction_encoding(),
            "goal_progress": self._completed.float(),
        }
        if self.oracle_observation:
            obs["oracle_unavailable"] = self._unavailable().float()
        if "state" in self.obs_mode:
            obs.update({
                "red_cube_pose": self.red_cube.pose.raw_pose,
                "blue_cube_pose": self.blue_cube.pose.raw_pose,
                "red_goal_pos": self.red_goal.pose.p,
                "blue_goal_pos": self.blue_goal.pose.p,
                "red_sweeper_pose": self.red_sweeper.pose.raw_pose,
                "blue_sweeper_pose": self.blue_sweeper.pose.raw_pose,
                "protected_pose": self.protected.pose.raw_pose,
            })
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        unavailable = self._unavailable()
        rows = torch.arange(self.num_envs, device=self.device)
        first = self._instruction_first
        second = 1 - first
        first_resolved = self._completed[rows, first] | unavailable[rows, first]
        active = torch.where(first_resolved, second, first)
        cube_positions = torch.stack([self.red_cube.pose.p, self.blue_cube.pose.p], dim=1)
        goal_positions = torch.stack([self.red_goal.pose.p, self.blue_goal.pose.p], dim=1)
        cube = cube_positions[rows, active]
        goal = goal_positions[rows, active]
        reaching = 1 - torch.tanh(5 * torch.linalg.norm(cube - self.agent.tcp.pose.p, dim=1))
        grasped = torch.where(
            active == 0, self.agent.is_grasping(self.red_cube), self.agent.is_grasping(self.blue_cube)
        )
        placing = 1 - torch.tanh(5 * torch.linalg.norm(cube - goal, dim=1))
        reward = reaching + grasped.float() * (1.0 + 2.0 * placing)
        reward += 3.0 * self._completed.float().sum(dim=1)
        reward -= 5.0 * self._constraint_violated.float()
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 10.0
