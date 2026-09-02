#!/usr/bin/env python3
"""Characterize the failures a nominal Peg policy produces on its own.

No intervention is applied. The policy runs the official contact-rich side
insertion and fails at its natural rate; this script records the terminal
physical state of every episode so the failure modes can be counted rather
than assumed.

The question this answers is whether emergent failures are *diverse*. A single
dominant mode poses no routing problem and cannot support a recovery
benchmark; several modes requiring different responses can.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.peg_insertion_recovery  # noqa: F401
from train_manipulation_ppo import Agent as StateAgent


def classify(ever_grasped, closest_lateral, grasp_steps, inserted):
    """Name a failure from the trajectory, not the terminal state.

    A first pass keyed on terminal state put every failure in one class, because
    every failed episode ends with the peg on the table. What separates the
    modes is the history: whether the peg was ever grasped, how close it got to
    the hole, and how long the grasp was held.
    """
    if inserted:
        return "success"
    if not ever_grasped:
        return "never_grasped"
    if closest_lateral > 0.08:
        return "grasp_lost_in_transport"
    if closest_lateral > 0.03:
        return "released_near_hole"
    return "reached_hole_then_lost"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-id", default="PegInsertionRecovery-v1")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=421_900_000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(
        args.env_id, num_envs=args.num_envs, reconfiguration_freq=1,
        max_episode_steps=args.steps, obs_mode="state",
        control_mode="pd_joint_delta_pos",
        # Intervention machinery is disabled, but the observation width depends
        # on how many intervention types are configured, so this must match the
        # competence evaluator for the nominal checkpoint to load.
        intervention_probability=0.0,
        intervention_types=("positive_lateral_peg_ejection",),
        # Drops the blocker dimensions the nominal policy was never trained
        # with; without this the observation is 50-wide and the checkpoint is
        # 43-wide. Matches the frozen competence_env_kwargs.
        include_blocker_state_observation=False,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True,
                             record_metrics=False)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    probe, _ = env.reset(seed=args.seed_base)
    observation_dim = int(probe.shape[1])
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = StateAgent(observation_dim, action_dim).to(device)
    agent.load_state_dict(checkpoint["agent"], strict=True)
    agent.eval()

    modes, episodes_done = Counter(), 0
    batches = max(1, args.episodes // args.num_envs)
    with torch.inference_mode():
        for batch in range(batches):
            obs, _ = env.reset(seed=args.seed_base + batch * args.num_envs)
            success = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            base = env.unwrapped
            ever_grasped = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            grasp_steps = torch.zeros(args.num_envs, device=device)
            closest = torch.full((args.num_envs,), 1e3, device=device)
            for _ in range(args.steps):
                action = agent.get_action(obs, deterministic=True).clamp(-1, 1)
                obs, _, _, _, info = env.step(action)
                success |= info["success"].bool()
                holding = base.agent.is_grasping(base.peg).bool()
                ever_grasped |= holding
                grasp_steps += holding.float()
                lateral = torch.linalg.norm(
                    (base.peg.pose.p - base.box_hole_pose.p)[:, :2], dim=1,
                )
                closest = torch.minimum(closest, lateral)

            done = success.cpu().numpy()
            held = ever_grasped.cpu().numpy()
            near = closest.cpu().numpy()
            steps_held = grasp_steps.cpu().numpy()
            for i in range(args.num_envs):
                modes[classify(bool(held[i]), float(near[i]),
                               float(steps_held[i]), bool(done[i]))] += 1
                episodes_done += 1

    env.close()
    total_failures = episodes_done - modes.get("success", 0)
    report = {
        "schema_version": 1,
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "intervention_probability": 0.0,
        "episodes": episodes_done,
        "successes": modes.get("success", 0),
        "failures": total_failures,
        "failure_rate": total_failures / episodes_done if episodes_done else None,
        "terminal_mode_counts": dict(modes),
        "failure_mode_share": {
            k: v / total_failures for k, v in modes.items()
            if k != "success" and total_failures
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["terminal_mode_counts"], indent=2))
    print(f"failure rate {report['failure_rate']:.4f} over {episodes_done} episodes")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
