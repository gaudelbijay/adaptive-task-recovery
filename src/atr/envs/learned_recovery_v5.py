"""Recovery benchmark whose ejection direction is not legible from one frame.

`LearnedRecovery-v4` implements forward and reverse ejection with *separate
actors*: a forward sweeper and a reverse sweeper, one per cube. Identifying the
mechanism therefore reduces to noticing which actor is displaced, and the
shortcut audit confirms it -- a single-observation model and a hand-written
motion threshold both reach the recurrent model's held-out accuracy exactly.
The benchmark was not testing composition; it was testing actor lookup.

This environment removes that affordance. Both ejection directions are produced
by *one* ejector per cube, which:

1. approaches the cube identically in both variants during an approach phase,
2. then, after a per-episode delay drawn from a range, receives a lateral force
   whose sign determines the direction.

Early frames are therefore identical in distribution across the two directions,
and the delay varies so no fixed decision step is correct. Distinguishing them
requires accumulating evidence after the divergence, which is the property that
the permanent/temporary pair already had and that ejection lacked.

Everything else -- task, reward, safety constraint, blockage mechanisms,
observation contract -- is inherited unchanged so results remain comparable to
`LearnedRecovery-v4`. No intervention assigns a pose; direction is produced by
forces alone.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from atr.envs.learned_recovery_v4 import (
    EJECTION, PERMANENT_BLOCK, REVERSE_EJECTION, TEMPORARY_BLOCK,
    LearnedRecoveryMechanismDiverseEnv,
)


@register_env("LearnedRecovery-v5", max_episode_steps=240)
class LearnedRecoveryDeferredDirectionEnv(LearnedRecoveryMechanismDiverseEnv):
    """Mechanism-diverse recovery with a direction-deferred shared ejector."""

    ejector_half_sizes = (0.025, 0.042, 0.025)

    def __init__(
        self,
        *args,
        direction_delay_range: tuple[int, int] = (10, 34),
        approach_force: float = 4.0,
        lateral_force: float = 6.0,
        **kwargs,
    ):
        self.direction_delay_range = (
            int(direction_delay_range[0]), int(direction_delay_range[1]),
        )
        self.approach_force = float(approach_force)
        self.lateral_force = float(lateral_force)
        if self.direction_delay_range[0] < 1:
            raise ValueError("direction delay must be at least one step")
        if self.direction_delay_range[0] >= self.direction_delay_range[1]:
            raise ValueError("direction delay range must be non-empty")
        self._direction_delay = None
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # One ejector per cube, used for *both* directions. The v4 forward and
        # reverse sweepers remain loaded so the observation contract and the
        # blockage mechanisms are unchanged; they are simply never driven here.
        self.red_ejector = actors.build_box(
            self.scene, self.ejector_half_sizes, [0.35, 0.35, 0.35, 1],
            "red_ejector", initial_pose=sapien.Pose(p=[0.30, -0.12, 0.025]),
        )
        self.blue_ejector = actors.build_box(
            self.scene, self.ejector_half_sizes, [0.35, 0.35, 0.35, 1],
            "blue_ejector", initial_pose=sapien.Pose(p=[0.30, 0.12, 0.025]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        low, high = self.direction_delay_range
        delay = torch.randint(
            low, high, (self.num_envs,), device=self.device, dtype=torch.long,
        )
        if self._direction_delay is None:
            self._direction_delay = delay
        else:
            self._direction_delay[env_idx] = delay[env_idx]

    def _before_simulation_step(self):
        # Inherit the blockage mechanisms unchanged while suppressing the v4
        # ejections. Both v4 sweeps are scaled by `intervention_force`, and the
        # blocker servo is scaled by `blocker_force`, so zeroing the former for
        # the duration of the inherited call drives the blockers exactly as
        # before and leaves the forward/reverse sweepers inert.
        inherited_force = self.intervention_force
        self.intervention_force = 0.0
        try:
            super()._before_simulation_step()
        finally:
            self.intervention_force = inherited_force
        step = self._episode_step
        mechanism = self._intervention_mechanism
        ejecting = (
            (step >= self._onset_step)
            & (step < self._onset_step + self.intervention_steps)
            & ((mechanism == EJECTION) | (mechanism == REVERSE_EJECTION))
        )
        if not bool(ejecting.any()):
            return

        diverged = step >= (self._onset_step + self._direction_delay)
        # +1 for forward, -1 for reverse, applied only after divergence so the
        # approach phase carries no directional information.
        sign = torch.where(
            mechanism == REVERSE_EJECTION,
            torch.full_like(self._direction_delay, -1.0, dtype=torch.float32),
            torch.full_like(self._direction_delay, 1.0, dtype=torch.float32),
        )

        for ejector, target in (
            (self.red_ejector, self._intervention_target == 0),
            (self.blue_ejector, self._intervention_target == 1),
        ):
            active = ejecting & target
            force = torch.zeros((self.num_envs, 3), device=self.device)
            # Approach is identical for both directions.
            force[:, 0] = self.approach_force * active.float()
            # Direction is applied laterally only once the delay has elapsed.
            force[:, 1] = (
                self.lateral_force * sign * (active & diverged).float()
            )
            self._apply_batched_force(ejector, force)

