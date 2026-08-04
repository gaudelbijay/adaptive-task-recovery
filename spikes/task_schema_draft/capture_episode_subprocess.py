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
"""

import argparse

import gymnasium as gym
import numpy as np

import task_schema_draft  # noqa: F401  (registers the env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=0, help="zero-action steps before capture")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--scene-variant", type=str, default="kitchen_cabinet",
        help='"kitchen_cabinet" (original) or "kitchen_sink" (D-027)',
    )
    args = parser.parse_args()

    env = gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind="chef_can_destroyed", onset_step_range=(2, 3),
        scene_variant=args.scene_variant,
    )
    env.reset(seed=args.seed)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(args.steps):
        env.step(zero_action)
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
