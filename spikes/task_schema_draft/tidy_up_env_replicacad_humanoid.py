"""TidyUp with a Unitree G1 humanoid PLACED (not navigating) in the real
ReplicaCAD apartment — the direct answer to "but this is not a humanoid
robot" for the ReplicaCAD variant. Same goal_graph.py / oracle_feasibility.py
/ intent_guard.py, unchanged.

G1 (`unitree_g1_simplified_upper_body_with_head_camera`) is fixed-base —
legs locked, cannot walk (see ../README.md "G1 in the real apartment").
So unlike tidy_up_env_replicacad.py's Fetch variant, this doesn't navigate:
G1 is placed once, at a spot with real open floor clearance (checked via
raycast, not guessed — see README), close enough to reach two nearby real
YCB objects with its right arm. Objects further away (bowl, cracker box)
are only used as monitored constraint targets, never touched — reach isn't
needed for privileged-state monitoring.

`ReplicaCADSceneBuilder.initialize()` only implements robot placement for
`robot_uids == "fetch"` and raises `NotImplementedError` for anything else
(checked directly, not assumed — it's a single hardcoded line). Everything
*before* that line (object placement for this episode) already ran
successfully by the time it raises, so catching it and placing G1 manually
is sufficient — no need to reimplement the scene setup.

Because G1 can't navigate, goal/constraint roles are swapped relative to
the Fetch variant: `potted_meat_can` and `master_chef_can` (both near the
chosen standing spot) are the goals here; `master_chef_can` is the one
destroyed by the intervention. The bowl — the Fetch variant's destroyed
object — is too far for this fixed-base robot to ever interact with, so
destroying it wouldn't demonstrate anything (it was already permanently
unreachable, not newly infeasible).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import sapien
import torch

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from task_schema_draft.goal_graph import Constraint, Goal, GoalGraph
from task_schema_draft.oracle_feasibility import ObjectState, WorldState, evaluate_goal_graph

_OBJECT_ALIASES = {
    "potted_meat_can": "env-0_010_potted_meat_can-1",
    "master_chef_can": "env-0_002_master_chef_can-2",
    "bowl": "env-0_024_bowl-4",
    "cracker_box": "env-0_003_cracker_box-1",
}

# Last-known world positions (ReplicaCADSetTableTrain seed=0), for objects
# that may no longer exist when a policy tries to reach them.
_LAST_KNOWN_POSITIONS = {
    "potted_meat_can": np.array([0.29, 0.09, 0.68]),
    "master_chef_can": np.array([0.81, 0.37, 0.71]),
}

# Base position chosen for having real open floor clearance nearby (checked
# via raycast: only 2 of 12 directional rays hit anything within 0.5m, the
# most open of five candidates tried — see README) and for having both goal
# objects within ~0.3m of the chosen standing spot, within this arm's
# reachable envelope.
_G1_BASE_POSE = [0.55, 0.23, 0.755]

# Reach targets don't need to be precise -- a successful attempt teleports
# the object onto the tray regardless of exact final tcp position, same
# abstraction as the other two variants (see policy_baselines_replicacad_humanoid.py).
_TRAY_POSITION = np.array([0.55, 0.23, 0.7])
_TRAY_HALF_SIZES = (0.25, 0.25, 0.15)
_REACH_CONFIGS = {
    "potted_meat_can": {
        "right_shoulder_pitch_joint": 0.9, "right_elbow_pitch_joint": 1.3,
        "right_shoulder_roll_joint": 0.3, "right_shoulder_yaw_joint": -0.8,
    },
    "master_chef_can": {
        "right_shoulder_pitch_joint": 0.9, "right_elbow_pitch_joint": 1.3,
        "right_shoulder_roll_joint": 0.3, "right_shoulder_yaw_joint": 0.8,
    },
}
_NEUTRAL_QPOS = np.zeros(25, dtype=np.float32)


def replicacad_humanoid_example() -> GoalGraph:
    """Same schema shape as goal_graph.canonical_example() / replicacad_example()
    — two goals, one never-move constraint, one maintain-orientation
    constraint — instantiated on real objects reachable from _G1_BASE_POSE."""
    return GoalGraph(
        instruction_text=(
            "Put the potted meat can and the master chef can on the tray, "
            "keep the cracker box upright, and do not move the bowl."
        ),
        goals=(
            Goal(id="place_can", predicate="on_tray", target_object="potted_meat_can", priority=0),
            Goal(id="place_chef_can", predicate="on_tray", target_object="master_chef_can", priority=0),
        ),
        constraints=(
            Constraint(
                id="keep_cracker_box_upright", kind="maintain_orientation",
                target_object="cracker_box", tolerance=0.85,
            ),
            Constraint(
                id="dont_move_bowl", kind="never_move",
                target_object="bowl", tolerance=0.05,
            ),
        ),
    )


class TidyUpReplicaCADHumanoidEnv(SceneManipulationEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]

    @property
    def _default_human_render_camera_configs(self):
        # SceneManipulationEnv's own default camera is tuned for its usual
        # fetch/panda setup and doesn't frame G1's fixed standing spot well
        # (confirmed by inspecting a rendered frame — mostly ceiling/floor).
        pose = sapien_utils.look_at([1.3, -0.3, 1.6], [0.55, 0.23, 0.8])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def __init__(
        self,
        *args,
        intervention_kind: Literal["chef_can_destroyed", "temporary_obstacle", "none"] = "chef_can_destroyed",
        onset_step_range: tuple[int, int] = (4, 6),
        obstacle_duration_steps: int = 10,
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self.obstacle_duration_steps = obstacle_duration_steps
        self.goal_graph = replicacad_humanoid_example()
        self._exists: dict[str, bool] = {}
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step: int | None = None
        self._initial_state: WorldState | None = None
        super().__init__(
            *args, robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            scene_builder_cls="ReplicaCADSetTableTrain", **kwargs,
        )

    def _get_actor(self, alias: str):
        return self.scene.actors[_OBJECT_ALIASES[alias]]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "TidyUpReplicaCADHumanoidEnv requires CPU sim — object "
                "add/remove is unsupported under GPU-batched sim."
            )
        # ReplicaCADRearrangeSceneBuilder.initialize() does real object
        # placement in TWO passes: objects go to a temporary pose+1000m-up
        # position first, THEN (after teleporting the robot, which is where
        # the fetch-only check lives) get moved to their real final pose in
        # a second pass. Catching NotImplementedError there — the obvious
        # thing to try — silently skips that second pass, leaving every
        # object floating ~1000m in the air (found by inspecting actual
        # positions after using that approach, not assumed). The actual fix:
        # temporarily present as "fetch" so the builder completes its own
        # full, correct placement logic, then set G1's real pose afterward.
        real_robot_uids = self.robot_uids
        self.robot_uids = "fetch"
        added_rest_keyframe = "rest" not in self.agent.keyframes
        if added_rest_keyframe:
            self.agent.keyframes["rest"] = self.agent.keyframes["standing"]
        try:
            super()._initialize_episode(env_idx, options)
        finally:
            self.robot_uids = real_robot_uids
            if added_rest_keyframe:
                del self.agent.keyframes["rest"]
        with torch.device(self.device):
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(sapien.Pose(p=_G1_BASE_POSE))

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
        if self.intervention_kind == "chef_can_destroyed":
            self._get_actor("master_chef_can").remove_from_scene()
            self._exists["master_chef_can"] = False
            # Removal alone doesn't refresh the render scene graph -- every
            # existing consumer of this env reads privileged state, not
            # pixels, so this went unnoticed until vision.py needed real
            # rendered frames to reflect the removal (see vision.py).
            self.scene.update_render()
        elif self.intervention_kind == "temporary_obstacle":
            from mani_skill.utils.building.actors import build_box

            near = self._get_actor("potted_meat_can").pose.sp.p
            self._obstacle = build_box(
                self.scene, half_sizes=[0.03, 0.03, 0.06], color=[0.3, 0.3, 0.3, 1],
                name="distractor_obstacle", body_type="static",
                initial_pose=sapien.Pose(p=near + np.array([0.1, 0.0, 0.06])),
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


@register_env("TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1", max_episode_steps=200)
class TidyUpReplicaCADHumanoidRegisteredEnv(TidyUpReplicaCADHumanoidEnv):
    pass
