"""Run the ManiSkill3 humanoid simulator-selection spike and record results.

See spikes/maniskill_humanoid_spike/README.md for what this is and isn't
testing: whether ManiSkill3 loads a humanoid asset, runs a scripted seeded
push deterministically, and performs adequately on this dev machine (no
CUDA). This is a spike, not a trained standing controller — the "hold"
baseline just targets the (noisy) initial standing pose for the episode.

Usage:
    python scripts/run_maniskill_humanoid_spike.py --robot g1 --episodes 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run(robot: str, episodes: int, max_steps: int, video_dir: Path, results_path: Path):
    import gymnasium as gym

    import maniskill_humanoid_spike  # noqa: F401  (registers HumanoidStandSpike-*-v1)
    from mani_skill.utils.wrappers.record import RecordEpisode

    env_id = {"g1": "HumanoidStandSpike-G1-v1", "h1": "HumanoidStandSpike-H1-v1"}[robot]

    results = {"env_id": env_id, "runs": []}

    for label, push_force_range in [
        ("baseline_no_push", (0.0, 0.0)),
        ("with_push", (80.0, 400.0)),
    ]:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend="cpu",
            push_force_range=push_force_range,
        )
        env = RecordEpisode(
            env,
            output_dir=str(video_dir / label),
            save_video=True,
            video_fps=30,
            trajectory_name=f"{env_id}_{label}",
            save_trajectory=False,
            max_steps_per_video=max_steps,
        )

        for ep in range(episodes):
            seed = 1000 + ep
            obs, info = env.reset(seed=seed)
            hold_action = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy()

            t0 = time.perf_counter()
            steps_survived = 0
            fell_step = None
            for step_idx in range(max_steps):
                obs, reward, terminated, truncated, info = env.step(hold_action)
                steps_survived += 1
                if not bool(info["is_standing"].item()) and fell_step is None:
                    fell_step = step_idx
                if bool(terminated.item()) or bool(truncated.item()):
                    break
            elapsed = time.perf_counter() - t0

            event = env.unwrapped.last_intervention_event
            results["runs"].append(
                {
                    "label": label,
                    "seed": seed,
                    "steps_survived": steps_survived,
                    "max_steps": max_steps,
                    "fell": fell_step is not None,
                    "fell_at_step": fell_step,
                    "steps_per_sec": round(steps_survived / elapsed, 1),
                    "intervention_event": None
                    if event is None
                    else {
                        "onset_step": event.onset_step,
                        "severity": round(float(event.severity), 4),
                        "force_magnitude_N": round(float(np.linalg.norm(event.force)), 2),
                    },
                }
            )
        env.close()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    return results


def summarize(results: dict):
    by_label: dict[str, list[dict]] = {}
    for run_ in results["runs"]:
        by_label.setdefault(run_["label"], []).append(run_)

    print(f"\n=== {results['env_id']} spike summary ===")
    for label, runs in by_label.items():
        n = len(runs)
        n_fell = sum(r["fell"] for r in runs)
        avg_steps = sum(r["steps_survived"] for r in runs) / n
        avg_sps = sum(r["steps_per_sec"] for r in runs) / n
        print(f"[{label}] {n} episodes, {n_fell}/{n} fell, "
              f"avg steps survived {avg_steps:.0f}, avg {avg_sps:.0f} steps/sec")
        for r in runs:
            ev = r["intervention_event"]
            ev_str = (
                f"push@step{ev['onset_step']} severity={ev['severity']} "
                f"{ev['force_magnitude_N']}N"
                if ev
                else "no push"
            )
            print(f"  seed={r['seed']} steps_survived={r['steps_survived']} "
                  f"fell={r['fell']}({r['fell_at_step']}) {ev_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["g1", "h1"], default="g1")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--video-dir", type=Path, default=Path("results/videos/maniskill_humanoid_spike")
    )
    parser.add_argument(
        "--results-path", type=Path, default=Path("results/maniskill_humanoid_spike.json")
    )
    args = parser.parse_args()

    results = run(args.robot, args.episodes, args.max_steps, args.video_dir, args.results_path)
    summarize(results)
