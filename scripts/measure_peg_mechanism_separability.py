#!/usr/bin/env python3
"""How far apart do PegInsertion's mechanisms actually move the peg?

Rendering the four mechanisms under the nominal controller produces four
episodes that look nearly identical. This measures whether that is a rendering
artifact or a property of the benchmark, by tracking peg displacement from its
own start for each mechanism on a shared seed.

It matters for how the audit's verdict is read. PegInsertion clears the ladder,
but a benchmark whose mechanisms are close to physically indistinguishable will
clear it too, and clearing means "the held-out mechanism is not shortcut-solved"
rather than "the benchmark is well posed".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_manipulation_ppo import Agent  # noqa: E402

import mani_skill.envs  # noqa: F401,E402
import atr.envs.peg_insertion_recovery  # noqa: F401,E402

KINDS = (
    "positive_lateral_peg_ejection",
    "negative_lateral_peg_ejection",
    "permanent_hole_block",
    "temporary_hole_block",
)


def blocker_flag(paths) -> bool:
    flags = set()
    for path in paths:
        task = torch.load(path, map_location="cpu", weights_only=False)["task"]
        kwargs = task.get("competence_env_kwargs", task.get("env_kwargs", {}))
        flags.add(bool(kwargs.get("include_blocker_state_observation", True)))
    if len(flags) != 1:
        raise SystemExit("nominal checkpoints disagree on the blocker observation contract")
    return next(iter(flags))


def trajectory(kind: str, args, flag: bool) -> np.ndarray:
    env = gym.make(
        "PegInsertionRecovery-v1", num_envs=args.num_envs, reconfiguration_freq=1,
        max_episode_steps=args.steps + 60, obs_mode="state", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", onset_step_range=(18, 42),
        intervention_probability=1.0, intervention_types=(kind,),
        include_blocker_state_observation=flag,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True,
                             record_metrics=False)
    observation, _ = env.reset(seed=args.seed)
    base = env.unwrapped
    agents = []
    for path in args.nominal_checkpoint:
        checkpoint = torch.load(path, map_location=base.device, weights_only=False)
        agent = Agent(int(np.prod(env.single_observation_space.shape)),
                      int(np.prod(env.single_action_space.shape))).to(base.device)
        agent.load_state_dict(checkpoint["agent"], strict=True)
        agent.eval()
        agents.append(agent)

    start = base.peg.pose.p.clone()
    displacement = []
    for _ in range(args.steps):
        with torch.no_grad():
            action = torch.stack([a.get_action(observation, True) for a in agents]).mean(0)
        observation, *_ = env.step(action)
        displacement.append((base.peg.pose.p - start)[args.index].cpu().numpy().copy())
    env.close()
    return np.asarray(displacement)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("results/peg_confusion_pair/mechanism_separability_v1.json"))
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--seed", type=int, default=511_000_000)
    args = parser.parse_args()

    flag = blocker_flag(args.nominal_checkpoint)
    traj = {kind: trajectory(kind, args, flag) for kind in KINDS}

    # The ejection pair is the contrast the audit turns on: on the tabletop
    # benchmark these two are trivially separable and a memoryless model reads
    # them perfectly.
    pair = np.linalg.norm(traj[KINDS[0]] - traj[KINDS[1]], axis=1)
    report = {
        "schema_version": 1,
        "protocol": "peg displacement from its own start, per mechanism, shared seed",
        "claim_boundary": (
            "Describes how far the mechanisms move the peg, not how well any "
            "policy responds to them."
        ),
        "steps": args.steps,
        "seed": args.seed,
        "ejection_pair_separation_max_m": float(pair.max()),
        "ejection_pair_separation_argmax_step": int(pair.argmax()),
        "ejection_pair_separation_final_m": float(pair[-1]),
        "final_displacement_m": {k: [float(x) for x in v[-1]] for k, v in traj.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"ejection pair: max {pair.max():.4f} m at step {int(pair.argmax())}, "
          f"final {pair[-1]:.4f} m")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
