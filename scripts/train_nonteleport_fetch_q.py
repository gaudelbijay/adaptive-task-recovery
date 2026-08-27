#!/usr/bin/env python3
"""Checkpointed reward training using the physical Fetch executor."""

from __future__ import annotations

import argparse

import gymnasium as gym

import task_schema_draft  # noqa: F401
from atr.envs.tidy_up_replicacad_manipulation import attempt_goal_with_real_grasp
from atr.envs.tidy_up_replicacad_policies import _TRAY_SLOTS
from atr.physical_pipeline import instruction_graph
from atr.policies.q_learning import train_q_table


def make_env(intervention_kind, onset_step_range):
    env = gym.make(
        "TidyUp-ReplicaCAD-v1", num_envs=1, obs_mode="state", render_mode=None,
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pos",
        intervention_kind=intervention_kind, onset_step_range=onset_step_range,
    )
    env.unwrapped.goal_graph = instruction_graph()
    return env


def normalized_physical_attempt(env, goal, tray_slot, nav_steps):
    """Execute real physics but normalize the bandit failure cost.

    The shared learner's historical ``-0.1 * steps_used`` reward assumed a
    25-step abstract reach.  A valid Fetch pick/carry/place takes roughly
    479 control steps, making one stochastic miss outweigh dozens of
    successes.  One failed committed skill is one -0.1 decision penalty;
    raw physical steps remain untouched in evaluation records.
    """
    result = attempt_goal_with_real_grasp(env, goal, tray_slot, nav_steps)
    return {**result, "physical_steps_used": result["steps_used"], "steps_used": 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()
    q = train_q_table(
        make_env=make_env, graph=instruction_graph(), tray_slots=_TRAY_SLOTS,
        attempt_goal_fn=normalized_physical_attempt,
        intervention_kinds=("cracker_box_destroyed", "cracker_box_destroyed"),
        onset_step_bounds=(5, 5), reach_steps=250, n_episodes=args.episodes,
        seed=args.seed, checkpoint_dir=args.checkpoint_dir, checkpoint_every=1,
    )
    print(q)


if __name__ == "__main__":
    main()
