#!/usr/bin/env python3
"""Verify rendered OOD profiles change pixels but not reset task state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import gymnasium as gym
import torch

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v3  # noqa: F401
import atr.envs.learned_recovery_v3_ood  # noqa: F401

from atr.envs.learned_recovery_v3_ood import PROFILES


STATE_KEYS = (
    "instruction", "goal_progress", "critic_red_cube_pose",
    "critic_blue_cube_pose", "critic_red_goal_pos", "critic_blue_goal_pos",
    "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
    "critic_protected_pose",
)


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=970013)
    parser.add_argument("--num-envs", type=int, default=4)
    args = parser.parse_args()
    records = []
    for profile in ("nominal", *PROFILES):
        kwargs = {
            "num_envs": args.num_envs,
            "reconfiguration_freq": 1,
            "obs_mode": "rgb",
            "render_mode": None,
            "sim_backend": "physx_cuda",
            "control_mode": "pd_joint_delta_pos",
            "vision_camera_size": 64,
            "asymmetric_critic_observation": True,
            "required_goals": 2,
            "intervention_probability": 0.5,
        }
        env_id = "LearnedRecovery-v3"
        if profile != "nominal":
            env_id = "LearnedRecovery-v3-OOD"
            kwargs["visual_domain_profile"] = profile
        env = gym.make(env_id, **kwargs)
        observation, _ = env.reset(seed=args.seed)
        rgb = observation["sensor_data"]["base_camera"]["rgb"]
        state = torch.cat([
            observation["extra"][key].float().reshape(args.num_envs, -1)
            for key in STATE_KEYS
        ] + [
            observation["agent"]["qpos"].float().reshape(args.num_envs, -1),
            observation["agent"]["qvel"].float().reshape(args.num_envs, -1),
        ], dim=1)
        records.append({
            "profile": profile, "environment": env_id,
            "rgb_sha256": tensor_sha256(rgb),
            "state_sha256": tensor_sha256(state),
            "rgb_shape": list(rgb.shape),
        })
        env.close()
    rgb_hashes = [record["rgb_sha256"] for record in records]
    state_hashes = [record["state_sha256"] for record in records]
    checks = {
        "all_renderings_distinct": len(set(rgb_hashes)) == len(rgb_hashes),
        "task_state_byte_identical": len(set(state_hashes)) == 1,
        "expected_profiles": [record["profile"] for record in records]
        == ["nominal", *PROFILES],
    }
    payload = {
        "schema_version": 1,
        "protocol": "rendered visual-OOD profile preflight",
        "seed": args.seed, "num_envs": args.num_envs,
        "checks": checks, "passed": all(checks.values()), "records": records,
        "source_sha256": {
            "preflight": file_sha256(Path(__file__)),
            "ood_environment": file_sha256(
                Path(__file__).resolve().parents[1]
                / "src/atr/envs/learned_recovery_v3_ood.py"
            ),
        },
        "claim_boundary": (
            "Reset-only rendering preflight: proves pixel changes and state "
            "invariance, not closed-loop robustness."
        ),
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
