"""Open a live SAPIEN viewer window and watch the humanoid-stand spike run.

Unlike run_maniskill_humanoid_spike.py (which renders to video files
headlessly), this opens an interactive GUI window on screen — you can orbit
the camera with the mouse while it runs. Loops through episodes back to back
until you close the window or hit Ctrl+C.

Usage:
    python scripts/watch_humanoid_stand_spike.py --robot g1
"""

from __future__ import annotations

import argparse
import time

import maniskill_humanoid_spike  # noqa: F401  (registers HumanoidStandSpike-*-v1)
import gymnasium as gym


def main(robot: str, episodes: int, push_onset_step_range: tuple[int, int], max_steps: int):
    env_id = {"g1": "HumanoidStandSpike-G1-v1", "h1": "HumanoidStandSpike-H1-v1"}[robot]
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        render_mode="human",
        sim_backend="cpu",
        push_onset_step_range=push_onset_step_range,
    )

    print(f"Opening viewer for {env_id} — close the window or Ctrl+C to stop.")
    try:
        for ep in range(episodes):
            seed = 2000 + ep
            env.reset(seed=seed)
            hold_action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()
            print(f"episode {ep} (seed={seed})")

            for step_idx in range(max_steps):
                env.step(hold_action)
                env.render()
                time.sleep(1 / 30)

                event = env.unwrapped.last_intervention_event
                if event is not None and event.onset_step == step_idx:
                    print(f"  push applied at step {step_idx}: "
                          f"severity={event.severity:.2f}")
            time.sleep(0.5)  # brief pause between episodes so the fall is visible
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["g1", "h1"], default="g1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--push-onset-range", type=int, nargs=2, default=[15, 25],
        help="control-step range the push can trigger in (kept early+narrow so you see it every episode)",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    main(args.robot, args.episodes, tuple(args.push_onset_range), args.max_steps)
