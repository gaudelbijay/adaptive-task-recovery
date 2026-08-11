"""Standalone script, run as a subprocess (never imported): runs exactly
one real full-agent episode -- `atr.pipeline.run_end_to_end_episode()`, real
CLIP perception plus a trained Q-table decision plus real arm motion -- and
writes its result to a JSON file, then exits.

Exists for the same reason as `capture_episode_subprocess.py` (D-052):
D-022's confirmed, open, unfixed upstream ManiSkill3 rendering-desync bug
means every render-producing reset in this project either stays within the
verified-safe budget (~2 per process) or gets its own fresh process.
`run_end_to_end_episode()` already keeps to exactly 2 renders (one per
goal) within its own single reset, so one episode per subprocess is always
"the first" from the OS's point of view -- the property a real multi-seed
benchmark (D-088) needs across N episodes, one subprocess per seed.

Usage: python run_full_agent_episode_subprocess.py --seed N --q-table-path PATH --out PATH
"""

import argparse
import json

import gymnasium as gym

import task_schema_draft  # noqa: F401  (registers all four envs)
from atr.evaluation.logging import _jsonable
from atr.pipeline import run_end_to_end_episode


def load_q_table(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {
        (entry["goal_id"], entry["feasible"]): {int(k): v for k, v in entry["actions"].items()}
        for entry in raw
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--intervention-kind", type=str, default="chef_can_destroyed")
    parser.add_argument("--scene-variant", type=str, default="kitchen_cabinet")
    parser.add_argument("--onset-step-min", type=int, default=1)
    parser.add_argument("--onset-step-max", type=int, default=3)
    parser.add_argument("--q-table-path", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    q_table = load_q_table(args.q_table_path)

    env = gym.make(
        "TidyUp-ReplicaCAD-Humanoid-v1", num_envs=1, obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", control_mode="pd_joint_pos",
        intervention_kind=args.intervention_kind,
        onset_step_range=(args.onset_step_min, args.onset_step_max),
        scene_variant=args.scene_variant,
    )
    try:
        env.reset(seed=args.seed)
        result = run_end_to_end_episode(env, q_table, scene_variant=args.scene_variant)
    finally:
        env.close()

    with open(args.out, "w") as f:
        json.dump(_jsonable(result), f)


if __name__ == "__main__":
    main()
