#!/usr/bin/env python3
"""Physics smoke for the direction-deferred ejection environment.

Checks the four conditions frozen in
`configs/learned_recovery_v5_deferred_direction.json`, plus the property the
design exists for: that displacement *before* the direction force is applied
carries no information about which direction is coming.

That last measurement is the point of the environment. If early displacement
already separates the two directions, the shared ejector has not removed the
affordance and the design does not work, regardless of the other checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v5  # noqa: F401

FORWARD, REVERSE = 0, 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path,
                        default=Path("configs/learned_recovery_v5_deferred_direction.json"))
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=140)
    # The first revision passed `--steps` as max_episode_steps and then ran
    # exactly that many steps, so truncation fired and every cube was restored
    # to its reset pose before being measured. Displacement was then identically
    # zero and direction correctness identically 0.0, which is what rejected the
    # environment twice. The horizon must outlast the measurement.
    parser.add_argument("--max-episode-steps", type=int, default=240)
    parser.add_argument("--direction-axis", type=int, default=0,
                        help="0 for axial (x), 1 for lateral (y).")
    parser.add_argument("--seed", type=int, default=511_000_000)
    parser.add_argument("--early-offset", type=int, default=6,
                        help="Steps after onset at which to sample early displacement.")
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text())["physics_smoke_gate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    early, late, mechanisms, delays = [], [], [], []
    collateral = []
    for batch in range(args.batches):
        env = gym.make(
            "LearnedRecovery-v5", num_envs=args.num_envs, reconfiguration_freq=1,
            max_episode_steps=args.max_episode_steps, obs_mode="state",
            control_mode="pd_joint_delta_pos",
            intervention_probability=1.0,
            intervention_types=("ejection", "reverse_ejection"),
        )
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True,
                                 record_metrics=False)
        env.reset(seed=args.seed + batch * args.num_envs)
        base = env.unwrapped

        def cube_positions():
            return torch.stack([base.red_cube.pose.p, base.blue_cube.pose.p], dim=1)

        start = cube_positions().clone()
        onset = base._onset_step.clone()
        delays.append(base._direction_delay.clone().cpu().numpy())
        mechanisms.append(base._intervention_mechanism.clone().cpu().numpy())
        target = base._intervention_target.clone()
        rows = torch.arange(args.num_envs, device=device)
        early_snapshot = None

        zero_action = torch.zeros(
            (args.num_envs,) + env.single_action_space.shape, device=device,
        )
        for step in range(1, args.steps + 1):
            env.step(zero_action)
            if early_snapshot is None and bool((step >= onset + args.early_offset).all()):
                early_snapshot = (cube_positions() - start)[rows, target].clone()
        final = (cube_positions() - start)[rows, target]
        other = (cube_positions() - start)[rows, 1 - target]

        early.append(early_snapshot.cpu().numpy())
        late.append(final.cpu().numpy())
        collateral.append(other.cpu().numpy())
        env.close()

    early = np.concatenate(early)
    late = np.concatenate(late)
    collateral = np.concatenate(collateral)
    mechanism = np.concatenate(mechanisms)
    delay = np.concatenate(delays)

    is_forward = mechanism == FORWARD
    is_reverse = mechanism == REVERSE
    moved = np.linalg.norm(late, axis=1) > 0.02
    # Direction is carried on `--direction-axis`.
    axis = args.direction_axis
    direction_correct = np.where(is_forward, late[:, axis] > 0, late[:, axis] < 0)

    def separability(displacement):
        """Fraction correctly classified by the sign of lateral displacement.

        0.5 means the two directions are indistinguishable at that point.
        """
        guess = displacement[:, axis] > 0
        return float(np.mean(guess == is_forward))

    report = {
        "schema_version": 1,
        "episodes": int(len(mechanism)),
        "observed_ejection_rate": float(moved.mean()),
        "direction_correctness": float(direction_correct[moved].mean()) if moved.any() else 0.0,
        "collateral_target_loss": float((np.linalg.norm(collateral, axis=1) > 0.02).mean()),
        "delay_distinct_values": int(len(np.unique(delay))),
        "early_direction_separability": separability(early),
        "late_direction_separability": separability(late),
        "early_offset_steps": args.early_offset,
        "direction_axis": axis,
        "max_episode_steps": args.max_episode_steps,
    }
    report["checks"] = {
        "observed_ejection_rate": report["observed_ejection_rate"] >= gate["minimum_observed_ejection_rate"],
        "direction_correctness": report["direction_correctness"] >= gate["minimum_direction_correctness"],
        "collateral_target_loss": report["collateral_target_loss"] <= gate["maximum_collateral_target_loss"],
        "delay_distinct_values": report["delay_distinct_values"] >= gate["minimum_delay_distinct_values"],
    }
    report["smoke_gate_pass"] = all(report["checks"].values())
    report["design_works"] = bool(
        report["early_direction_separability"] < 0.65
        and report["late_direction_separability"] > 0.90
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
