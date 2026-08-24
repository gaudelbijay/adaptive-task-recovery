"""Standalone script, run as a subprocess (never imported): captures exactly
one render-producing reset of TidyUpReplicaCADHumanoidEnv, then exits.

Exists because of D-022 (ai-notes/decisions.md) -- a confirmed, open, unfixed
upstream ManiSkill3 bug (haosulab/ManiSkill#1150) where rendered frames from
this env desync from the actual scene after roughly the second
render-producing reset in one process. Every render-producing reset in this
project either stays within that verified-safe budget (clip_feasibility.py's tests: at
most 2, in one process, per test run) or -- as here, for dinov2_probe.py,
which needs more than 2 labeled examples -- gets its own fresh process, so
each capture is always "the first" from the OS's point of view.

Usage: python capture_episode_subprocess.py --seed N --steps K --out path.npz
Saves: frame (uint8 HWC), and exists_master_chef_can / exists_potted_meat_can
(bool) to the given .npz path. `--steps` uses zero actions (the same
hold-the-keyframe-pose pattern every other file in this project uses, e.g.
`env.action_space.sample() * 0`) -- not random actions: this env's
`pd_joint_pos` action space is raw joint-angle targets (checked directly:
Box bounds like [-2.618, 2.9671], not normalized [-1, 1]), so random actions
risk unrealistic/unstable arm motion for no real benefit. `--steps` controls
only how many zero-action steps run before capture, i.e. whether the
scripted intervention (onset step 2) has fired yet by capture time.

`--attempt-object` (added D-055, ai-notes/decisions.md) optionally runs a
real *full* attempt_goal() -- reach, then teleport-to-tray if the object
exists -- for the named object before capture, so a labeled example
includes both G1's arm having moved into frame *and* whatever else that
first attempt visibly changed in the scene (e.g. the first object now
sitting in the tray). Matters because a reach-only capture turned out not
to reproduce D-054's actual failure (see D-055) -- the live pipeline's
first goal doesn't just move the arm, it also (when successful) moves that
object into the tray, and that's part of what the second goal's frame
looks like too. `--intervention-kind` (default unchanged,
"chef_can_destroyed") lets a caller ask for "none" instead, needed to
collect an arm-in-frame *present* example (nothing destroyed) to pair
against the arm-in-frame *absent* one.
"""

import argparse

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401  (registers the env)
from atr.language.goal_graph import Goal
from atr.envs.tidy_up_replicacad_humanoid_policies import _TRAY_SLOTS, attempt_goal

# Matches _instruction_graph()'s goal order in atr.pipeline (parsed from
# replicacad_humanoid_example()'s instruction text: potted meat can first,
# then master chef can) -- not imported directly, since atr.pipeline sits
# above atr.envs and this script must not depend downward on it.
_GOAL_ORDER = ["potted_meat_can", "master_chef_can"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=0, help="zero-action steps before capture")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--scene-variant", type=str, default="kitchen_cabinet",
        help='"kitchen_cabinet", "kitchen_sink" (D-027), or "third_layout" (D-121)',
    )
    parser.add_argument(
        "--intervention-kind", type=str, default="chef_can_destroyed",
        help='env intervention_kind, e.g. "chef_can_destroyed" or "none"',
    )
    parser.add_argument(
        "--attempt-object", type=str, default=None, choices=_GOAL_ORDER,
        help="if set, perform a real attempt_goal() (reach + teleport-on-success) for "
        "this object before capture -- puts G1's arm (and, if successful, the object "
        "itself) in frame the way an actual prior goal attempt would, D-055",
    )
    args = parser.parse_args()

    env = gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind=args.intervention_kind, onset_step_range=(2, 3),
        scene_variant=args.scene_variant,
    )
    env.reset(seed=args.seed)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(args.steps):
        env.step(zero_action)
    if args.attempt_object is not None:
        goal = Goal(id="_capture", predicate="on_tray", target_object=args.attempt_object, priority=0)
        tray_slot = _TRAY_SLOTS[_GOAL_ORDER.index(args.attempt_object)]
        attempt_goal(env, goal, tray_slot)
    frame = env.render()[0].cpu().numpy()
    exists = dict(env.unwrapped._exists)
    env.close()

    np.savez(
        args.out, frame=frame,
        exists_master_chef_can=exists["master_chef_can"],
        exists_potted_meat_can=exists["potted_meat_can"],
    )


if __name__ == "__main__":
    main()
