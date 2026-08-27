#!/usr/bin/env python3
"""Capture a real rendered episode from a frozen manipulation checkpoint."""

from __future__ import annotations

import argparse
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


def _success(info: dict, env) -> bool:
    candidates = []
    for key in ("success_once", "success_at_end", "success"):
        if key in info:
            candidates.append(info[key])
    final_info = info.get("final_info")
    if isinstance(final_info, dict):
        episode = final_info.get("episode", {})
        for key in ("success_once", "success_at_end", "success"):
            if key in episode:
                candidates.append(episode[key])
    if not candidates:
        candidates.append(env.unwrapped.evaluate().get("success", False))
    for value in candidates:
        if isinstance(value, torch.Tensor):
            if bool(value.detach().cpu().bool().any().item()):
                return True
        elif bool(np.asarray(value).any()):
            return True
    return False


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/manipulation_ppo_v1.json")
    parser.add_argument("--results", default="results/manipulation_ppo")
    parser.add_argument("--output", default="results/manipulation_ppo/videos")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--seed-base", type=int, default=81000000)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("policy capture requires a CUDA GPU")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    task, _ = _select_task(config, args.task_index)
    training_seed = int(task["seed"])
    run_dir = Path(args.results) / config["name"] / task["env_id"] / f"seed_{training_seed}"
    checkpoint_path = run_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    if checkpoint["task"] != task:
        raise ValueError("checkpoint task does not match immutable task configuration")

    env_kwargs = _environment_kwargs(task)
    env_kwargs["render_mode"] = "rgb_array"
    env = gym.make(task["env_id"], num_envs=1, reconfiguration_freq=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, record_metrics=True)
    observation_dim = int(np.prod(env.single_observation_space.shape))
    action_dim = int(np.prod(env.single_action_space.shape))
    agent = Agent(observation_dim, action_dim).cuda()
    agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    action_low = torch.as_tensor(env.single_action_space.low, device="cuda")
    action_high = torch.as_tensor(env.single_action_space.high, device="cuda")

    selected = None
    with torch.no_grad():
        for attempt in range(args.max_attempts):
            episode_seed = args.seed_base + args.task_index * 100000 + attempt
            observation, _ = env.reset(seed=episode_seed)
            frames = [_frame(env)]
            success_once = False
            steps = 0
            for steps in range(1, int(task["num_eval_steps"]) + 1):
                action = torch.clamp(agent.get_action(observation, deterministic=True), action_low, action_high)
                observation, _, terminated, truncated, info = env.step(action)
                success_once = success_once or _success(info, env)
                if bool((terminated | truncated).reshape(-1)[0].item()):
                    break
                frames.append(_frame(env))
            if success_once:
                selected = (episode_seed, steps, frames)
                break
    env.close()
    if selected is None:
        raise RuntimeError(f"no successful episode found in {args.max_attempts} declared held-out seeds")

    episode_seed, steps, frames = selected
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = task["env_id"].removesuffix("-v1")
    video_path = output_dir / f"{stem}.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=2)
    metadata = {
        "schema_version": 1,
        "env_id": task["env_id"],
        "training_seed": training_seed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "episode_seed": episode_seed,
        "steps": steps,
        "frames": len(frames),
        "fps": args.fps,
        "success_once": True,
        "protocol": "deterministic frozen-policy visualization from declared held-out seed range",
    }
    _atomic_json(metadata, output_dir / f"{stem}.json")
    print(json.dumps({**metadata, "video": str(video_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
