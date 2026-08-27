#!/usr/bin/env python3
"""Capture declared recovery branches from a frozen learned-control policy."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from train_manipulation_ppo import Agent, _environment_kwargs, _select_task


def _frame(env) -> np.ndarray:
    image = env.render()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image.astype(np.uint8, copy=False)


def _scalar(info: dict, key: str, default: bool = False) -> bool:
    value = info.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learned_recovery_ppo_v6.json")
    parser.add_argument("--results", default="results/learned_recovery")
    parser.add_argument("--output", default="results/learned_recovery/videos")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--branch", choices=("first_goal_removed", "second_goal_removed", "nominal"),
        required=True,
    )
    parser.add_argument("--seed-base", type=int, default=91000000)
    parser.add_argument("--max-attempts", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("policy capture requires a CUDA GPU")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = _select_task(config, args.task_index)
    if task.get("registration_module"):
        importlib.import_module(task["registration_module"])
    method = task.get("method", task["env_id"])
    training_seed = int(task["seed"])
    run_dir = Path(args.results) / config["name"] / method / f"seed_{training_seed}"
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")

    env_kwargs = _environment_kwargs(task, evaluation=True)
    env_kwargs["render_mode"] = "rgb_array"
    env_kwargs["intervention_probability"] = 0.0 if args.branch == "nominal" else 1.0
    env = gym.make(task["env_id"], num_envs=1, reconfiguration_freq=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    # Keep the terminal success state renderable. The wrapper otherwise
    # autoresets immediately and the final video frame belongs to a new task.
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)
    observation_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = Agent(observation_dim, action_dim).cuda()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()

    def rollout(episode_seed: int, capture: bool = False):
        observation, _ = env.reset(seed=episode_seed)
        frames = [_frame(env)] if capture else []
        branch_matches = args.branch == "nominal"
        success_once = False
        violation_once = False
        steps = 0
        for steps in range(1, int(task["num_eval_steps"]) + 1):
            observation, _, _, truncated, info = env.step(
                agent.get_action(observation, deterministic=True)
            )
            if steps == 1 and args.branch != "nominal":
                first_removed = _scalar(info, "first_goal_removed")
                branch_matches = first_removed == (args.branch == "first_goal_removed")
            success_once |= _scalar(info, "success")
            violation_once |= _scalar(info, "constraint_violated")
            if capture:
                frames.append(_frame(env))
            if success_once or violation_once or _scalar({"done": truncated}, "done"):
                break
        return branch_matches, success_once, violation_once, steps, frames

    selected = None
    with torch.no_grad():
        for attempt in range(args.max_attempts):
            episode_seed = args.seed_base + attempt
            branch_matches, success_once, violation_once, steps, _ = rollout(episode_seed)
            if branch_matches and success_once and not violation_once:
                selected = (episode_seed, steps)
                break
    if selected is None:
        env.close()
        raise RuntimeError(
            f"no safe successful {args.branch} episode found in "
            f"{args.max_attempts} declared seeds beginning at {args.seed_base}"
        )

    episode_seed, search_steps = selected
    with torch.no_grad():
        branch_matches, success_once, violation_once, steps, frames = rollout(
            episode_seed, capture=True
        )
    env.close()
    if not branch_matches or not success_once or violation_once or steps != search_steps:
        raise RuntimeError("deterministic rendered replay did not match the qualifying search rollout")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{method}_{training_seed}_{args.branch}"
    video_path = output_dir / f"{stem}.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=2)
    metadata = {
        "schema_version": 1,
        "protocol": "first safe success in a declared sequential seed range",
        "env_id": task["env_id"],
        "method": method,
        "branch": args.branch,
        "training_seed": training_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "seed_base": args.seed_base,
        "max_attempts": args.max_attempts,
        "episode_seed": episode_seed,
        "steps": steps,
        "frames": len(frames),
        "fps": args.fps,
        "safe_success": True,
        "teleport_calls": 0,
    }
    _atomic_json(metadata, output_dir / f"{stem}.json")
    print(json.dumps({**metadata, "video": str(video_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
