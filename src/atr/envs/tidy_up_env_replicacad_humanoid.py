"""TidyUp with a Unitree G1 humanoid PLACED (not navigating) in the real
ReplicaCAD apartment — the direct answer to "but this is not a humanoid
robot" for the ReplicaCAD variant. Same goal_graph.py / oracle_feasibility.py
/ intent_guard.py, unchanged.

Promoted to src/atr/ 2026-08-04 (D-049) -- see ai-notes/decisions.md.
Registered as `TidyUp-ReplicaCAD-Humanoid-v1` (was
`TidyUpTaskSchemaDraft-ReplicaCAD-Humanoid-v1`).

G1 (`unitree_g1_simplified_upper_body_with_head_camera`) is fixed-base —
legs locked, cannot walk (see spikes/task_schema_draft/README.md "G1 in
the real apartment").
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

Scene layout is reproducible across seeds, not literally pinned by index --
a real bug found by D-020's vision work, then a real correction to how it
was understood to be fixed found by D-119. `ReplicaCADRearrangeSceneBuilder`
samples the apartment layout (`build_config_idxs`) and, separately, an init
variant (`init_config_idxs`) that determines which subset of YCB objects
are actually placed versus hidden at z=-10000 -- both driven by torch's
*global* RNG state, not this env's own `_episode_rng`. Every existing test
of this env (D-018) only ever called `reset(seed=0)`, so this was
invisible: `env.reset(seed=2)` loads a different apartment entirely
(confirmed by rendering it -- G1 ends up next to a couch and a bicycle),
and even pinning `build_config_idxs` alone still leaves the "which objects
are hidden" draw seed-dependent (confirmed: some seeds hide
`master_chef_can` or `potted_meat_can` outright, at z=-10000, with
build_config_idxs held fixed). G1's base pose, camera, and
clip_feasibility.py's crop regions are all calibrated for one specific
resulting layout -- the one `reset(seed=0)` happened to produce before this
fix existed.

D-021's original fix constructor-pins `build_config_idxs` and
*attempts* to pin `init_config_idxs` (`_SCENE_INIT_CONFIG_IDX`, passed as a
constructor kwarg) alongside explicitly reseeding torch's global RNG
(`torch.manual_seed(_SCENE_TORCH_SEED)`) immediately before both scene
constructions call into the scene builder. **D-119 found the
`init_config_idxs` half of that never actually worked**: instrumented
`ReplicaCADRearrangeSceneBuilder.sample_init_config_idxs()` directly and
confirmed it *is* called (returning `[10]` for `kitchen_cabinet`, not the
constructor's `[0]`) -- `SceneManipulationEnv.reset()`, ManiSkill3's own
base class, unconditionally discards the constructor's `init_config_idxs`
on the first (reconfiguring) reset every env instance makes. Reproducibility
across `reset(seed=...)` calls comes entirely from the `torch.manual_seed`
reseed making that *sampled* draw deterministic, not from the constructor
value being honored -- both shipped layouts (`kitchen_cabinet`/
`kitchen_sink`) were tuned by searching `_SCENE_TORCH_SEED` values for one
that samples a good init config, not by choosing an init config index
directly. Do not trust `_SCENE_INIT_CONFIG_IDX`'s literal value for
anything; it is kept only so removing it isn't a separate, riskier change
(see its own definition below). This env's other stochastic element
(intervention onset step) is unaffected by any of this -- it's drawn from
`self._episode_rng` (numpy, seeded from the `reset(seed=...)` argument
through ManiSkill's own machinery), a separate stream from torch's global
RNG.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import sapien
import torch

from mani_skill.envs.scenes.base_env import SceneManipulationEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from atr.language.goal_graph import Constraint, Goal, GoalGraph
from atr.feasibility.oracle import ObjectState, WorldState, evaluate_goal_graph

_OBJECT_ALIASES = {
    "potted_meat_can": "env-0_010_potted_meat_can-1",
    "master_chef_can": "env-0_002_master_chef_can-2",
    "bowl": "env-0_024_bowl-4",
    "cracker_box": "env-0_003_cracker_box-1",
}

# D-119: R-014/D-061's real mechanism -- confirmed by instrumenting
# ReplicaCADRearrangeSceneBuilder directly, not inferred -- is that
# `_SCENE_INIT_CONFIG_IDX` below is never actually used: ManiSkill3's own
# `SceneManipulationEnv.reset()` discards the constructor's `init_config_idxs`
# on every reconfiguring reset (`self.init_config_idxs = options.get(...,
# None)`, unconditionally None on that branch) and falls back to
# `sample_init_config_idxs()` -- confirmed by patching that method to record
# calls: it *is* invoked, returning `[10]` for kitchen_cabinet, not `[0]`.
# Reproducibility comes entirely from `torch.manual_seed(_SCENE_TORCH_SEED)`
# right before that sample, not from the constructor value -- matching the
# older module-docstring comment about "searching torch seed values 0-14",
# which was the real mechanism D-020/D-021 used, not index pinning.
#
# This means `_OBJECT_ALIASES`' hardcoded instance suffixes were tuned
# against whatever object layout a *given (build_config_idx, torch seed)*
# pair happens to sample -- not a literal, inspectable init config index.
# Verified directly (not assumed) that for `master_chef_can`/`potted_meat_can`
# /`bowl`, the instance closest to the scene's own `base_pose` is exactly the
# one `_OBJECT_ALIASES` already hardcodes, in both existing layouts -- so
# `_resolve_dynamic_actor_name()` below reproduces the current, working
# aliases exactly (zero behavior change) while removing the fragility a
# hardcoded suffix has if this env is ever pointed at a different
# build_config/seed pair. `cracker_box` is the one exception found: its
# nearest instance is NOT the one `_OBJECT_ALIASES` hardcodes, in both
# layouts, and no rule that reproduces the hardcoded choice was found -- left
# hardcoded rather than shipping an unverified guess for it specifically.
_DYNAMIC_ALIAS_OBJECT_IDS = {
    "potted_meat_can": "010_potted_meat_can",
    "master_chef_can": "002_master_chef_can",
    "bowl": "024_bowl",
}
_CRACKER_BOX_OBJECT_ID = "003_cracker_box"

# D-121: the two scenes `_OBJECT_ALIASES`'s hardcoded `cracker_box` suffix
# was tuned against. A brand-new `build_config_idx` has its own, unrelated
# YCB-instance numbering (same D-119 lesson, applied to the one object
# dynamic resolution didn't already cover) -- the hardcoded string is not
# just unverified for a new scene, it's essentially guaranteed wrong (wrong
# instance count, likely hidden or nonexistent). Any `scene_variant` outside
# this set also resolves `cracker_box` dynamically (see `_initialize_episode`)
# instead of falling back to a alias tuned for a different scene entirely.
_LEGACY_CALIBRATED_SCENE_VARIANTS = frozenset({"kitchen_cabinet", "kitchen_sink"})


def _resolve_dynamic_actor_name(scene, obj_id: str, base_xy, env_num: int = 0) -> str:
    """Among every built (possibly-hidden) instance of `obj_id`, return the
    non-hidden one closest to `base_xy` -- the property that, verified
    directly, reproduces every currently-shipped hardcoded alias this
    replaces. Raises if none are placed at all, rather than silently
    returning something wrong -- this project's established convention
    (see instruction_parser.py's module docstring) for an unresolvable
    reference."""
    best_name: str | None = None
    best_distance: float | None = None
    i = 0
    while True:
        name = f"env-{env_num}_{obj_id}-{i}"
        if name not in scene.actors:
            break
        position = scene.actors[name].pose.sp.p
        if position[2] > -100:  # not hidden at the scene builder's default z=-10000-ish
            distance = float(np.hypot(position[0] - base_xy[0], position[1] - base_xy[1]))
            if best_distance is None or distance < best_distance:
                best_name, best_distance = name, distance
        i += 1
    if best_name is None:
        raise ValueError(f"no placed (non-hidden) instance of {obj_id!r} found for env {env_num}")
    return best_name

# Last-known world positions (ReplicaCADSetTableTrain seed=0), for objects
# that may no longer exist when a policy tries to reach them.
_LAST_KNOWN_POSITIONS = {
    "potted_meat_can": np.array([0.29, 0.09, 0.68]),
    "master_chef_can": np.array([0.81, 0.37, 0.71]),
}

# The specific apartment layout / object arrangement G1's base pose, camera,
# and clip_feasibility.py's crop regions are all calibrated against -- see module
# docstring "Scene layout is pinned, not sampled." Found by recording what
# reset(seed=0) sampled before this fix existed (build_config_idx=59), then
# searching torch seed values (0-14) at that fixed build/init config for one
# that keeps both target objects actually placed (not hidden at z=-10000);
# seed 0 reproduces the exact original layout.
#
# D-027: a *second* calibrated layout ("kitchen_sink"), added specifically to
# answer the caveat that clip_feasibility.py/dinov2_probe.py were only ever
# validated on one scene. build_config_idx=55 chosen the same way as 59 --
# searched under the real class's actual two-pin torch.manual_seed pattern
# (D-021's own lesson: a naive single-pin search gives different, wrong
# results), for a config where both target objects are placed and close
# together. Only vision/rendering are recalibrated for it (base pose,
# camera, clip_feasibility.py's crops) -- reach configs and tray position are NOT,
# since arm-reach/grasp was never what this second layout was for; using
# "kitchen_sink" with the reach-dependent policy baselines is out of scope
# and untested.
_SCENE_CONFIGS = {
    "kitchen_cabinet": {
        "build_config_idx": 59,
        "base_pose": [0.55, 0.23, 0.755],
        "camera_eye": [1.3, -0.3, 1.6],
        "camera_target": [0.55, 0.23, 0.8],
    },
    "kitchen_sink": {
        "build_config_idx": 55,
        "base_pose": [-1.77, -1.03, 0.755],
        "camera_eye": [-1.3, -0.5, 2.4],
        "camera_target": [-2.0, -1.3, 0.85],
    },
    # D-121: third layout, found the way D-116 recommended -- checked real
    # placed positions directly (not a separate raycast/visual proxy that
    # turned out not to correspond to the same thing, D-061's actual
    # mistake) across all 61 other valid build_config_idx values at
    # torch_seed=0. Filtered on two independently checked properties, not
    # just one: master_chef_can/potted_meat_can proximity, then floor
    # clearance at the resulting midpoint -- several candidates that looked
    # good on the first (e.g. build_config_idx=31, objects only 0.14m apart)
    # turned out to have the standing spot fully enclosed (12/12 raycast
    # hits) once actually checked, confirming clearance can't be assumed
    # from object proximity alone. build_config_idx=17 has
    # master_chef_can/potted_meat_can 0.169m apart, zero of 12 raycast hits
    # at the midpoint (kitchen_cabinet's own base position only cleared
    # 10/12), and -- checked separately again -- both `bowl` and
    # `cracker_box` land within about 1.1m of that same midpoint too,
    # unlike most other candidates where one of those two ends up several
    # meters away. base_pose is that midpoint at the same standing height
    # (0.755) both other layouts use; camera reuses kitchen_cabinet's
    # relative eye/target offset as an untuned starting point -- confirmed
    # non-degenerate by a real subprocess-isolated render capture (D-022's
    # established pattern), not shipped unseen. Privileged-state only:
    # reach configs/tray position (reach was never real contact-range
    # reachability even for the calibrated layout, see ik_solver.py's
    # module docstring) are shared and object-keyed, not scene-specific, so
    # those need no per-scene entry -- but clip_feasibility.py's crops are
    # NOT calibrated for this layout, unlike kitchen_sink's.
    "third_layout": {
        "build_config_idx": 17,
        "base_pose": [0.781, -7.088, 0.755],
        "camera_eye": [1.531, -7.618, 1.600],
        "camera_target": [0.781, -7.088, 0.800],
    },
}
_SCENE_INIT_CONFIG_IDX = 0
_SCENE_TORCH_SEED = 0

# Guard for a real rendering bug (D-022): the *rendered* scene graph (not
# privileged state -- object positions stay correct, verified by
# test_scene_layout_reproducible_across_seeds) desyncs from the actual
# built scene after roughly the second render-producing reset/instantiation
# of this env within one process -- confirmed process-wide (also reproduces
# on tidy_up_env_replicacad.py, a different env class), confirmed
# independent of seed, reconfigure forcing, and sapien.render.clear_cache().
# Confirmed as a known, OPEN, upstream ManiSkill3 bug, not anything in this
# project's code: haosulab/ManiSkill#1150, macOS-only, YCB-object scenes,
# breaks after the 2nd/3rd reset -- exactly this. No maintainer fix exists.
# This counts render-producing resets in this process and warns past the
# point that's actually been empirically verified safe, so a silently-wrong
# render becomes a loud warning instead -- see clip_feasibility.py and README
# "Vision layer."
_render_producing_reset_count = 0
# Conservative: fresh-instantiation-each-time testing showed problems as
# early as the *second* render-producing reset in a process (same-instance
# reuse tolerated a bit more) -- warn from the second one onward rather
# than assume the more generous case.
_RENDER_SAFE_LIMIT = 1

# "kitchen_cabinet"'s base position was chosen for having real open floor
# clearance nearby (checked via raycast: only 2 of 12 directional rays hit
# anything within 0.5m, the most open of five candidates tried — see
# README) and for having both goal objects within ~0.3m of the chosen
# standing spot, within this arm's reachable envelope. "kitchen_sink"'s
# base position (D-027) was raycast-checked the same way but is NOT
# reach-calibrated -- see _SCENE_CONFIGS above.

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
    constraint — instantiated on real objects reachable from the
    "kitchen_cabinet" scene's base position (the only variant reach configs
    exist for)."""
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
        cfg = _SCENE_CONFIGS[self._scene_variant]
        pose = sapien_utils.look_at(cfg["camera_eye"], cfg["camera_target"])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def __init__(
        self,
        *args,
        intervention_kind: Literal["chef_can_destroyed", "temporary_obstacle", "none"] = "chef_can_destroyed",
        onset_step_range: tuple[int, int] = (4, 6),
        obstacle_duration_steps: int = 10,
        scene_variant: Literal["kitchen_cabinet", "kitchen_sink", "third_layout"] = "kitchen_cabinet",
        **kwargs,
    ):
        self.intervention_kind = intervention_kind
        self.onset_step_range = onset_step_range
        self.obstacle_duration_steps = obstacle_duration_steps
        self.goal_graph = replicacad_humanoid_example()
        self._scene_variant = scene_variant
        self._exists: dict[str, bool] = {}
        self._resolved_aliases: dict[str, str] = {}
        self._onset_step: int | None = None
        self._elapsed_control_steps = 0
        self._triggered = False
        self._obstacle = None
        self._obstacle_remove_step: int | None = None
        self._initial_state: WorldState | None = None
        # Force the calibrated layout regardless of other caller-supplied
        # values -- G1's base pose, camera, and clip_feasibility.py's crops are
        # meaningless on any other layout (see module docstring "Scene
        # layout is pinned"). Which of the two calibrated layouts is up to
        # scene_variant, not the caller's own build_config_idxs.
        kwargs["build_config_idxs"] = [_SCENE_CONFIGS[scene_variant]["build_config_idx"]]
        # D-119: this init_config_idxs value is not actually honored -- see
        # that decision's note above `_DYNAMIC_ALIAS_OBJECT_IDS` -- kept only
        # because removing it would be a separate, riskier behavior change
        # (it would make ManiSkill3 actually respect index 0, which would
        # very likely select a different real layout than the one G1's base
        # pose/camera/clip crops are calibrated against).
        kwargs["init_config_idxs"] = [_SCENE_INIT_CONFIG_IDX]
        super().__init__(
            *args, robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            scene_builder_cls="ReplicaCADSetTableTrain", **kwargs,
        )

    def _get_actor(self, alias: str):
        name = self._resolved_aliases.get(alias, _OBJECT_ALIASES.get(alias))
        return self.scene.actors[name]

    def _load_scene(self, options: dict):
        # Pins which apartment layout gets sampled -- see module docstring.
        torch.manual_seed(_SCENE_TORCH_SEED)
        super()._load_scene(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "TidyUpReplicaCADHumanoidEnv requires CPU sim — object "
                "add/remove is unsupported under GPU-batched sim."
            )
        # Bug guard (D-022): counts render-producing resets, not just
        # reconfigures -- the desync reproduces even across repeated
        # reset() calls on one already-built instance, which never re-runs
        # _load_scene (reconfiguration_freq defaults to 0). See the
        # constant's own comment above for what this actually guards.
        if self.render_mode is not None:
            global _render_producing_reset_count
            _render_producing_reset_count += 1
            if _render_producing_reset_count > _RENDER_SAFE_LIMIT:
                warnings.warn(
                    f"This is render-producing reset #{_render_producing_reset_count} of "
                    "TidyUpReplicaCADHumanoidEnv in this process. Rendered frames past "
                    "roughly the 2nd such reset have been observed to desync from the "
                    "actual scene (object positions stay correct; the image doesn't) -- "
                    "a known, open, unfixed upstream ManiSkill3 bug on macOS with "
                    "YCB-object scenes (haosulab/ManiSkill#1150), not fixable here. Do "
                    "not trust clip_feasibility.py results from this env instance without visually "
                    "verifying the frame first. See D-022 in ai-notes/decisions.md.",
                    stacklevel=2,
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
        #
        # Also pins which objects end up placed vs. hidden at z=-10000 --
        # a second, separate torch.randint draw inside initialize() that
        # build_config_idxs/init_config_idxs alone don't control (see module
        # docstring "Scene layout is pinned").
        torch.manual_seed(_SCENE_TORCH_SEED)
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
            self.agent.robot.set_pose(sapien.Pose(p=_SCENE_CONFIGS[self._scene_variant]["base_pose"]))

        base_xy = _SCENE_CONFIGS[self._scene_variant]["base_pose"][:2]
        self._resolved_aliases = {
            alias: _resolve_dynamic_actor_name(self.scene, obj_id, base_xy)
            for alias, obj_id in _DYNAMIC_ALIAS_OBJECT_IDS.items()
        }
        if self._scene_variant not in _LEGACY_CALIBRATED_SCENE_VARIANTS:
            # D-121: no hardcoded alias exists for cracker_box on a new scene
            # to preserve -- resolve it the same way as the other 3 rather
            # than fall back to a suffix tuned for a completely different
            # build_config_idx.
            self._resolved_aliases["cracker_box"] = _resolve_dynamic_actor_name(
                self.scene, _CRACKER_BOX_OBJECT_ID, base_xy
            )

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
            # pixels, so this went unnoticed until clip_feasibility.py needed real
            # rendered frames to reflect the removal (see clip_feasibility.py).
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


@register_env("TidyUp-ReplicaCAD-Humanoid-v1", max_episode_steps=200)
class TidyUpReplicaCADHumanoidRegisteredEnv(TidyUpReplicaCADHumanoidEnv):
    pass
