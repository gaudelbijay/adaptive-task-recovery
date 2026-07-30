"""TidyUp on a real ReplicaCAD apartment scene with a Fetch mobile robot —
the third embodiment/scene combination, after the panda-arm tabletop
(tidy_up_env.py) and the G1-humanoid kitchen counter
(tidy_up_env_humanoid.py). Same goal_graph.py / oracle_feasibility.py /
intent_guard.py, unchanged.

Per the user's request to prefer established environments over hand-built
ones: this reuses ManiSkill3's own `SceneManipulationEnv` +
`ReplicaCADSetTableTrain` scene builder — a real furnished apartment (104
actors) with real YCB objects (Habitat's rearrangement dataset), not
hand-placed primitive boxes. Real objects were inspected directly (not
assumed): `env-0_024_bowl-4` is a genuine YCB bowl, etc. — see
../README.md "ReplicaCAD embodiment" for the full object inventory used.

Important scope finding, also in ../README.md: these scenes scatter active
objects across the *entire apartment* (rooms 1-2+ meters apart), and use
`fetch`, a **mobile** base robot — not the fixed-arm/fixed-reach setup the
panda and humanoid versions use. So "attempt a goal" here genuinely
requires navigation before reaching, not just an arm swing. `_navigate_to`
implements a simple proportional go-to-pose controller (turn to face
target, then drive forward) using Fetch's `base` sub-controller
(`PDBaseForwardVelControllerConfig`: action = [forward_vel, turn_vel]).
Placement still uses the teleport-on-success abstraction, same reason as
the other two variants.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.utils.registration import register_env

from task_schema_draft.goal_graph import Constraint, Goal, GoalGraph
from task_schema_draft.oracle_feasibility import ObjectState, WorldState, evaluate_goal_graph

# Real actor names in ReplicaCADSetTableTrain's seed=0 build/init config,
# found by inspecting env.unwrapped.scene.actors directly (see README) --
# not guessed. Kept as an alias map so the goal graph reads with plain
# semantic names instead of "env-0_024_bowl-4".
_OBJECT_ALIASES = {
    "potted_meat_can": "env-0_010_potted_meat_can-1",
    "bowl": "env-0_024_bowl-4",
    "master_chef_can": "env-0_002_master_chef_can-2",
    "cracker_box": "env-0_003_cracker_box-1",
}


def replicacad_example() -> GoalGraph:
    """The same schema as goal_graph.canonical_example(), instantiated on
    real objects that actually exist in this scene (no "mug"/"glass" YCB
    model exists, so this isn't literally the docs/01 wording -- it's the
    same *structure*: two goals, one never-move constraint, one
    maintain-orientation constraint)."""
    return GoalGraph(
        instruction_text=(
            "Put the potted meat can and the bowl on the table, keep the "
            "cracker box upright, and do not move the master chef can."
        ),
        goals=(
            Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can", priority=0),
            Goal(id="place_bowl", predicate="on_tray", target_object="bowl", priority=0),
        ),
        constraints=(
            Constraint(
                id="keep_cracker_box_upright", kind="maintain_orientation",
                target_object="cracker_box", tolerance=0.85,
            ),
            Constraint(
                id="dont_move_master_chef_can", kind="never_move",
                target_object="master_chef_can", tolerance=0.05,
            ),
        ),
    )


_TRAY_POSITION = np.array([-1.0, 0.6, 0.7])
_TRAY_HALF_SIZES = (0.3, 0.3, 0.15)


class TidyUpReplicaCADEnv(SceneManipulationEnv):
    """Same interventions as the other two TidyUp variants, layered on top
    of a real ReplicaCAD apartment scene instead of a hand-built one."""

    SUPPORTED_ROBOTS = ["fetch"]

    def __init__(
        self,
        *args,
        intervention_kind: Literal["bowl_destroyed", "temporary_obstacle", "none"] = "bowl_destroyed",
        onset_step_range: tuple[int, int] = (2, 3),
        obstacle_duration_steps: int = 20,
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self.obstacle_duration_steps = obstacle_duration_steps
        self.goal_graph = replicacad_example()
        self._exists: dict[str, bool] = {}
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step: int | None = None
        self._initial_state: WorldState | None = None
        super().__init__(
            *args, robot_uids="fetch", scene_builder_cls="ReplicaCADSetTableTrain", **kwargs
        )

    def _get_actor(self, alias: str):
        return self.scene.actors[_OBJECT_ALIASES[alias]]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "TidyUpReplicaCADEnv requires CPU sim — object add/remove is "
                "unsupported under GPU-batched sim. Pass sim_backend='physx_cpu'."
            )
        super()._initialize_episode(env_idx, options)

        seed = int(self._episode_rng.randint(0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        self._onset_step = int(rng.integers(*self.onset_step_range))
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step = None
        self._exists = {alias: True for alias in _OBJECT_ALIASES}
        self._initial_state = None

    _SETTLE_STEPS = 3

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
            self._get_actor("bowl").remove_from_scene()
            self._exists["bowl"] = False
        elif self.intervention_kind == "temporary_obstacle":
            import sapien

            from mani_skill.utils.building.actors import build_box

            near_can = self._get_actor("potted_meat_can").pose.sp.p
            self._obstacle = build_box(
                self.scene, half_sizes=[0.05, 0.05, 0.1], color=[0.3, 0.3, 0.3, 1],
                name="distractor_obstacle", body_type="static",
                initial_pose=sapien.Pose(p=near_can + np.array([0.2, 0.0, 0.1])),
            )
            self.scene.update_render()
            self._obstacle_remove_step = self._elapsed_control_steps + self.obstacle_duration_steps

    def _world_state(self) -> WorldState:
        state: WorldState = {}
        for alias in _OBJECT_ALIASES:
            if not self._exists[alias]:
                state[alias] = ObjectState(exists=False, position=None, up_vector=None)
                continue
            pose = self._get_actor(alias).pose.sp
            up_vector = pose.to_transformation_matrix()[:3, 2]
            state[alias] = ObjectState(exists=True, position=pose.p.copy(), up_vector=up_vector.copy())
        return state

    def evaluate(self):
        current_state = self._world_state()
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

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)


@register_env("TidyUpTaskSchemaDraft-ReplicaCAD-v1", max_episode_steps=2000)
class TidyUpReplicaCADRegisteredEnv(TidyUpReplicaCADEnv):
    pass
