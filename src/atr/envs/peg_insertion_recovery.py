"""No-teleport recovery extension of ManiSkill3 ``PegInsertionSide-v1``.

The native peg, box, randomization, robot controller, dense reward, and success
predicate are inherited unchanged. Runtime perturbations act through forces:
two matched lateral impulses disturb the peg, while a dynamic blocker is driven
into the hole and either held there or retracted. Actor poses are assigned only
during episode initialization.

Privileged mechanism state is exposed for labels and scoring. It is forbidden
as a router input by the external publishability gate.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


INTERVENTION_TYPES = (
    "positive_lateral_peg_ejection",
    "permanent_hole_block",
    "temporary_hole_block",
    "negative_lateral_peg_ejection",
)
POSITIVE_EJECTION = 0
PERMANENT_BLOCK = 1
TEMPORARY_BLOCK = 2
NEGATIVE_EJECTION = 3


@register_env("PegInsertionRecovery-v1", max_episode_steps=160)
class PegInsertionRecoveryEnv(PegInsertionSideEnv):
    """Official side insertion with force-driven recovery interventions."""

    # Smaller than the narrowest randomized hole (radius >= 0.018 m), but
    # large enough to obstruct the native peg (radius >= 0.015 m).
    blocker_half_sizes = (0.014, 0.014, 0.014)

    def __init__(
        self,
        *args,
        intervention_probability: float = 0.8,
        intervention_types: Sequence[str] = INTERVENTION_TYPES,
        onset_step_range: Sequence[int] = (18, 42),
        ejection_force: float = 2.4,
        ejection_steps: int = 5,
        negative_ejection_force_scale: float = 0.85,
        blocker_force: float = 5.0,
        blocker_position_gain: float = 40.0,
        blocker_velocity_gain: float = 4.0,
        blocker_return_delay_steps: int = 48,
        **kwargs,
    ):
        kinds = tuple(intervention_types)
        unknown = set(kinds) - set(INTERVENTION_TYPES)
        if not kinds or unknown:
            raise ValueError(f"invalid intervention types: {sorted(unknown)}")
        if len(onset_step_range) != 2 or onset_step_range[0] > onset_step_range[1]:
            raise ValueError("onset_step_range must be an ordered pair")
        if not 0.0 <= intervention_probability <= 1.0:
            raise ValueError("intervention_probability must lie in [0, 1]")
        self.intervention_probability = float(intervention_probability)
        self.intervention_types = kinds
        self.onset_step_range = tuple(int(value) for value in onset_step_range)
        self.ejection_force = float(ejection_force)
        self.ejection_steps = int(ejection_steps)
        self.negative_ejection_force_scale = float(negative_ejection_force_scale)
        self.blocker_force = float(blocker_force)
        self.blocker_position_gain = float(blocker_position_gain)
        self.blocker_velocity_gain = float(blocker_velocity_gain)
        self.blocker_return_delay_steps = int(blocker_return_delay_steps)
        self._intervention_mechanism = None
        self._onset_step = None
        self._blocker_engaged = None
        self._temporary_cleared = None
        self._constraint_violated = None
        self._blocker_home = None
        self._blocker_target = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.hole_blocker = actors.build_sphere(
            self.scene,
            self.blocker_half_sizes[0],
            [0.18, 0.18, 0.18, 1.0],
            "hole_blocker",
            initial_pose=sapien.Pose(p=[0.34, -0.34, self.blocker_half_sizes[2]]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self._intervention_mechanism is None:
            with torch.device(self.device):
                self._intervention_mechanism = torch.full(
                    (self.num_envs,), -1, dtype=torch.long,
                )
                self._onset_step = torch.zeros(self.num_envs, dtype=torch.long)
                self._blocker_engaged = torch.zeros(self.num_envs, dtype=torch.bool)
                self._temporary_cleared = torch.zeros(self.num_envs, dtype=torch.bool)
                self._constraint_violated = torch.zeros(self.num_envs, dtype=torch.bool)
                self._blocker_home = torch.zeros((self.num_envs, 3))
                self._blocker_target = torch.zeros((self.num_envs, 3))
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            batch = len(env_idx)
            choices = torch.tensor(
                [INTERVENTION_TYPES.index(kind) for kind in self.intervention_types],
                dtype=torch.long,
            )
            selected = choices[torch.randint(0, len(choices), (batch,))]
            enabled = torch.rand(batch) < self.intervention_probability
            self._intervention_mechanism[env_idx] = torch.where(
                enabled, selected, torch.full_like(selected, -1),
            )
            low, high = self.onset_step_range
            self._onset_step[env_idx] = torch.randint(low, high + 1, (batch,))
            self._blocker_engaged[env_idx] = False
            self._temporary_cleared[env_idx] = False
            self._constraint_violated[env_idx] = False
            # Stage the blocker directly outside the randomized hole along
            # its local insertion axis. Its subsequent path is therefore
            # force-driven through the opening rather than diagonally into a
            # box wall. This pose assignment occurs only during reset.
            hole_raw = self.box_hole_pose.raw_pose[env_idx]
            hole_pose = Pose.create_from_pq(hole_raw[:, :3], hole_raw[:, 3:])
            local_target = torch.zeros((batch, 3))
            local_target[:, 0] = -(
                self.peg_half_sizes[env_idx, 0]
                + self.blocker_half_sizes[0]
                + 0.006
            )
            target = (hole_pose * Pose.create_from_pq(local_target)).p
            local_home = local_target.clone()
            # Five centimetres of force-driven travel is large relative to
            # the randomized hole diameter but short enough for a bounded
            # servo to converge uniformly across heterogeneous GPU scenes.
            local_home[:, 0] -= 0.05
            home = (hole_pose * Pose.create_from_pq(local_home)).p
            self._blocker_home[env_idx] = home
            self._blocker_target[env_idx] = target
            self.hole_blocker.set_pose(Pose.create_from_pq(home))
            self.hole_blocker.set_linear_velocity(torch.zeros((batch, 3)))
            self.hole_blocker.set_angular_velocity(torch.zeros((batch, 3)))

    def _apply_batched_force(self, actor, force: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            actor.apply_force(force)
        else:
            actor.apply_force(force[0].detach().cpu().numpy().astype(np.float32))

    def _before_simulation_step(self):
        step = self._elapsed_steps
        started = step >= self._onset_step
        ejection_active = started & (
            step < self._onset_step + self.ejection_steps
        )
        positive = self._intervention_mechanism == POSITIVE_EJECTION
        negative = self._intervention_mechanism == NEGATIVE_EJECTION
        peg_force = torch.zeros((self.num_envs, 3), device=self.device)
        peg_force[:, 1] = self.ejection_force * (
            (ejection_active & positive).float()
            - self.negative_ejection_force_scale
            * (ejection_active & negative).float()
        )
        self._apply_batched_force(self.peg, peg_force)

        permanent = self._intervention_mechanism == PERMANENT_BLOCK
        temporary = self._intervention_mechanism == TEMPORARY_BLOCK
        block_kind = permanent | temporary
        block_active = started & block_kind
        returning = temporary & (
            step >= self._onset_step + self.blocker_return_delay_steps
        )
        target = self._blocker_target
        home = self._blocker_home
        toward_hole = block_active & ~returning
        desired = torch.where(toward_hole[:, None], target, home)
        raw_force = (
            self.blocker_position_gain * (desired - self.hole_blocker.pose.p)
            - self.blocker_velocity_gain * self.hole_blocker.linear_velocity
        )
        norm = torch.linalg.vector_norm(raw_force, dim=1, keepdim=True).clamp_min(1e-6)
        bounded_force = raw_force * (self.blocker_force / norm).clamp(max=1.0)
        self._apply_batched_force(
            # The same bounded physical servo holds the unused blocker at its
            # reset staging pose before onset, so gravity cannot turn onset
            # timing into a geometry confound.
            self.hole_blocker, bounded_force,
        )

        target_distance = torch.linalg.vector_norm(
            self.hole_blocker.pose.p - target, dim=1,
        )
        home_distance = torch.linalg.vector_norm(
            self.hole_blocker.pose.p - home, dim=1,
        )
        # The native box has a nonzero collision envelope: a dynamic blocker
        # contacting the mouth stops before its center reaches the geometric
        # command point. The 6.5 cm bound was fixed by the force-only physics
        # audit and represents contact at the entrance, not penetration.
        self._blocker_engaged |= toward_hole & (target_distance < 0.065)
        self._temporary_cleared |= returning & (home_distance < 0.05)
        peg_position = self.peg.pose.p
        blocker_protected = self._blocker_engaged & ~self._temporary_cleared & block_kind
        blocker_clearance = self.blocker_half_sizes[0] + self.peg_half_sizes[:, 1]
        peg_head_blocker_distance = torch.linalg.vector_norm(
            self.peg_head_pose.p - self.hole_blocker.pose.p, dim=1,
        )
        self._constraint_violated |= (
            (peg_position[:, 2] < -0.02)
            | (torch.linalg.vector_norm(peg_position[:, :2], dim=1) > 0.8)
            | (blocker_protected & (peg_head_blocker_distance < blocker_clearance))
        )

    def evaluate(self):
        info = super().evaluate()
        mechanism = self._intervention_mechanism
        physical_unavailable = (mechanism == PERMANENT_BLOCK) & self._blocker_engaged
        intervention_finished = (
            (mechanism < 0)
            | (
                ((mechanism == POSITIVE_EJECTION) | (mechanism == NEGATIVE_EJECTION))
                & (self._elapsed_steps >= self._onset_step + self.ejection_steps)
            )
            | physical_unavailable
            | ((mechanism == TEMPORARY_BLOCK) & self._temporary_cleared)
        )
        info.update({
            "constraint_violated": self._constraint_violated,
            "critic_intervention_mechanism": mechanism,
            "critic_intervention_onset_step": self._onset_step,
            "critic_physical_unavailable": physical_unavailable,
            "blocker_engaged": self._blocker_engaged,
            "temporary_cleared": self._temporary_cleared,
            "intervention_finished": intervention_finished,
            "critic_blocker_target_distance": torch.linalg.vector_norm(
                self.hole_blocker.pose.p - self._blocker_target, dim=1,
            ),
            "critic_peg_head_blocker_distance": torch.linalg.vector_norm(
                self.peg_head_pose.p - self.hole_blocker.pose.p, dim=1,
            ),
            # Physical task geometry used by the state-based router. The
            # intervention identity and feasibility labels above are never
            # included in this tensor. Each 7-D block is position followed by
            # quaternion, matching ManiSkill's native state observation.
            "router_task_geometry": torch.cat((
                self.peg.pose.raw_pose,
                self.box_hole_pose.raw_pose,
                self.hole_blocker.pose.raw_pose,
                self.agent.tcp.pose.raw_pose,
            ), dim=1),
        })
        return info

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            obs["hole_blocker_pose"] = self.hole_blocker.pose.raw_pose
        return obs
