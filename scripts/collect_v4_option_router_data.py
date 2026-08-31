#!/usr/bin/env python3
"""Collect causal state prefixes for the learned V4 recovery router.

Evaluator-only mechanism fields create training targets but are never included
in a feature.  Every saved row is a trajectory prefix, and ``group_id`` keeps
all prefixes from one physical episode in the same statistical split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mani_skill.envs  # noqa: F401
import atr.envs.learned_recovery_v4  # noqa: F401
from train_visual_recovery_dual_teacher_ppo import (
    VisualAgent, extract_observation, observation_contract,
    privileged_aux_dim, select_task,
)


KINDS = ("nominal", "ejection", "permanent_block", "temporary_block", "reverse_ejection")
SNAPSHOTS = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96)
POSE_KEYS = (
    "critic_red_cube_pose", "critic_blue_cube_pose",
    "critic_red_sweeper_pose", "critic_blue_sweeper_pose",
    "critic_red_reverse_sweeper_pose", "critic_blue_reverse_sweeper_pose",
    "critic_red_goal_blocker_pose", "critic_blue_goal_blocker_pose",
)
FORBIDDEN_FEATURE_KEYS = (
    "critic_intervention_mechanism", "critic_goal_resolved",
    "intervention_mechanism", "intervention_target",
)


def make_env(num_envs: int, kind: str, *, onset: int, force_scale: float, return_delay: int):
    kwargs = dict(
        num_envs=num_envs, obs_mode="rgb", render_mode=None,
        sim_backend="physx_cuda", control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense", reconfiguration_freq=1,
        asymmetric_critic_observation=True,
        intervention_probability=0.0 if kind == "nominal" else 1.0,
        intervention_types=("ejection",) if kind == "nominal" else (kind,),
        onset_step_range=(onset, onset), intervention_force=6.0 * force_scale,
        intervention_steps=24, blocker_force=4.0 * force_scale,
        blocker_return_force=5.0 * force_scale,
        blocker_return_delay_steps=return_delay,
        terminate_on_violation=True, safety_proximity_weight=5.0,
        constraint_violation_penalty=20.0, vision_camera_size=64,
    )
    env = gym.make("LearnedRecovery-v4", **kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=False)


def feature_contract(obs) -> list[str]:
    names = []
    for key in POSE_KEYS:
        names.extend(f"{key}.goal_relative.{axis}" for axis in "xyz")
    for family in ("sweeper", "reverse_sweeper", "goal_blocker"):
        names.extend(
            f"{family}.cube_relative.{color}.{axis}"
            for color in ("red", "blue") for axis in "xyz"
        )
    for target in ("red_cube", "blue_cube", "red_goal", "blue_goal", "protected"):
        names.extend(f"tcp.{target}_relative.{axis}" for axis in "xyz")
    names.extend(f"instruction.{index}" for index in range(obs["extra"]["instruction"].shape[1]))
    names.extend(f"goal_progress.{index}" for index in range(obs["extra"]["goal_progress"].shape[1]))
    names.extend(f"agent.qpos.{index}" for index in range(obs["agent"]["qpos"].shape[1]))
    names.extend(f"agent.qvel.{index}" for index in range(obs["agent"]["qvel"].shape[1]))
    names.append("normalized_time")
    return names


def extract_features(
    obs, initial_positions, previous_positions, initial_tcp, initial_qpos,
    step: int, horizon: int,
):
    extra = obs["extra"]
    positions = torch.stack([extra[key][:, :3] for key in POSE_KEYS], dim=1)
    cube_positions = positions[:, :2]
    goals = torch.stack((extra["critic_red_goal_pos"], extra["critic_blue_goal_pos"]), dim=1)
    paired_goals = goals.repeat(1, 4, 1)
    actor_goal = positions - paired_goals
    mechanism_cube = positions[:, 2:] - cube_positions.repeat(1, 3, 1)
    tcp = extra["tcp_pose"][:, :3]
    protected = extra["critic_protected_pose"][:, :3]
    tcp_relative = torch.cat((
        (tcp[:, None] - cube_positions).flatten(1),
        (tcp[:, None] - goals).flatten(1),
        tcp - protected,
    ), dim=1)
    time = torch.full((positions.shape[0], 1), step / horizon, device=positions.device)
    feature = torch.cat((
        actor_goal.flatten(1), mechanism_cube.flatten(1), tcp_relative,
        extra["instruction"].float(), extra["goal_progress"].float(),
        obs["agent"]["qpos"].float(),
        obs["agent"]["qvel"].float(), time,
    ), dim=1)
    return feature, positions


def labels(kind: str, cleared: torch.Tensor, started: bool):
    count = len(cleared)
    option = torch.full((count,), {
        "nominal": 0, "ejection": 1, "reverse_ejection": 2,
        "permanent_block": 3, "temporary_block": 4,
    }[kind], dtype=torch.long, device=cleared.device)
    if kind == "temporary_block":
        option = torch.where(cleared.bool(), torch.zeros_like(option), option)
    event = torch.full((count,), {
        "nominal": 0, "ejection": 1, "reverse_ejection": 1,
        "permanent_block": 2, "temporary_block": 2,
    }[kind], dtype=torch.long, device=cleared.device)
    direction = torch.full((count,), -100, dtype=torch.long, device=cleared.device)
    if kind == "ejection": direction.fill_(0)
    if kind == "reverse_ejection": direction.fill_(1)
    block = torch.full((count,), -100, dtype=torch.long, device=cleared.device)
    if kind == "permanent_block": block.fill_(1)
    if kind == "temporary_block": block = torch.where(cleared.bool(), torch.full_like(block, 2), torch.zeros_like(block))
    if not started:
        event.zero_()
        direction.fill_(-100)
        block.fill_(-100)
    return option, event, direction, block


@torch.inference_mode()
def collect_kind(args, kind: str, kind_index: int, rows: dict[str, list], behavior):
    for batch in range(args.batches_per_kind):
        rng = np.random.default_rng(args.seed_base + kind_index * 100_000 + batch)
        onset = int(rng.integers(args.onset_min, args.onset_max + 1))
        force_scale = float(rng.uniform(args.force_scale_min, args.force_scale_max))
        return_delay = int(rng.integers(args.return_delay_min, args.return_delay_max + 1))
        env = make_env(args.num_envs, kind, onset=onset, force_scale=force_scale, return_delay=return_delay)
        try:
            episode_seed = args.seed_base + kind_index * 10_000_000 + batch * args.num_envs
            obs, info = env.reset(seed=episode_seed)
            positions = torch.stack([obs["extra"][key][:, :3] for key in POSE_KEYS], dim=1)
            initial_positions = positions.clone()
            previous_positions = positions.clone()
            initial_tcp = obs["extra"]["tcp_pose"][:, :3].clone()
            initial_qpos = obs["agent"]["qpos"].clone()
            history = []
            for step in range(1, args.horizon + 1):
                # Prefix time is the pre-action observation, exactly matching
                # deployment.  In particular, snapshot 1 is the reset state
                # and cannot contain evidence produced by the first action.
                feature, positions = extract_features(
                    obs, initial_positions, previous_positions, initial_tcp, initial_qpos,
                    step, args.horizon,
                )
                previous_positions = positions
                history.append(feature)
                if step in SNAPSHOTS and step <= args.horizon:
                    sequence = torch.stack(history, dim=1)
                    padded = torch.zeros(
                        args.num_envs, args.horizon, feature.shape[1],
                        dtype=feature.dtype, device=feature.device,
                    )
                    padded[:, :step] = sequence
                    cleared = info["temporary_block_cleared"].bool()
                    option, event, direction, block = labels(
                        kind, cleared, step >= onset + 2,
                    )
                    rows["sequence"].append(padded.cpu().numpy().astype(np.float32))
                    rows["length"].append(np.full(args.num_envs, step, dtype=np.int64))
                    rows["option"].append(option.cpu().numpy())
                    rows["event"].append(event.cpu().numpy())
                    rows["direction"].append(direction.cpu().numpy())
                    rows["block_status"].append(block.cpu().numpy())
                    rows["temporary_cleared"].append(cleared.cpu().numpy())
                    rows["onset"].append(np.full(args.num_envs, onset, dtype=np.int64))
                    rows["return_delay"].append(np.full(args.num_envs, return_delay, dtype=np.int64))
                    # The entire vectorized reset is one split unit: its actors
                    # share physics parameters and randomization draws.
                    rows["group_id"].append(np.full(
                        args.num_envs, kind_index * 1_000_000_000 + batch,
                        dtype=np.int64,
                    ))
                    rows["condition"].append(np.full(args.num_envs, kind_index, dtype=np.int64))
                if behavior is None:
                    action = torch.zeros(
                        (args.num_envs,) + env.single_action_space.shape,
                        dtype=torch.float32, device=positions.device,
                    )
                else:
                    agent, task = behavior
                    rgb, proprio, _ = extract_observation(
                        obs, task["asymmetric_critic"],
                        task.get("actor_tcp_pose", False),
                        task.get("actor_goal_progress", False),
                    )
                    latent = agent.encode(rgb)
                    progress = torch.sigmoid(agent.goal_progress_predictor(latent))
                    action = torch.tanh(agent.actor(torch.cat((latent, proprio, progress), dim=1)))
                obs, _, _, _, info = env.step(action)
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/router/v4_option_prefixes_train_v1.npz")
    parser.add_argument("--metadata-output", default="results/router/v4_option_prefixes_train_v1.json")
    parser.add_argument("--batches-per-kind", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed-base", type=int, default=310_000_000)
    parser.add_argument("--onset-min", type=int, default=0)
    parser.add_argument("--onset-max", type=int, default=8)
    parser.add_argument("--force-scale-min", type=float, default=0.85)
    parser.add_argument("--force-scale-max", type=float, default=1.15)
    parser.add_argument("--return-delay-min", type=int, default=24)
    parser.add_argument("--return-delay-max", type=int, default=36)
    parser.add_argument("--conditions", default=",".join(KINDS))
    parser.add_argument("--behavior", choices=("zero", "nominal"), default="zero")
    parser.add_argument("--policy-index", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--config", default="configs/visual_recovery_dual_specialist_dagger_v19.json")
    parser.add_argument("--checkpoint-root", default="results/visual_recovery_ppo/visual_recovery_dual_specialist_dagger_v19")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V4 collection requires CUDA")
    rows = {key: [] for key in (
        "sequence", "length", "option", "event", "direction",
        "block_status", "temporary_cleared", "onset", "return_delay",
        "group_id", "condition",
    )}
    selected_conditions = tuple(item.strip() for item in args.conditions.split(",") if item.strip())
    unknown = sorted(set(selected_conditions) - set(KINDS))
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    behavior = None
    behavior_checkpoint = None
    if args.behavior == "nominal":
        config = json.loads(Path(args.config).read_text())
        task, _ = select_task(config, args.policy_index)
        bootstrap = make_env(1, "nominal", onset=0, force_scale=1.0, return_delay=30)
        try:
            obs, _ = bootstrap.reset(seed=args.seed_base - 2)
            rgb, proprio, critic = extract_observation(
                obs, task["asymmetric_critic"],
                task.get("actor_tcp_pose", False),
                task.get("actor_goal_progress", False),
            )
            agent = VisualAgent(
                task["image_size"], proprio.shape[1], critic.shape[1],
                int(np.prod(bootstrap.single_action_space.shape)),
                task["asymmetric_critic"], task.get("augmentation_pad", 0),
                privileged_aux_dim(task), task.get("actor_learned_goal_progress", False),
            ).cuda()
            checkpoint_path = Path(args.checkpoint_root) / task["method"] / f"seed_{task['seed']}" / "best.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
            if checkpoint.get("observation_contract") != observation_contract(task):
                raise ValueError("nominal checkpoint observation contract mismatch")
            agent.load_state_dict(checkpoint["agent"])
            agent.eval()
            behavior = (agent, task)
            behavior_checkpoint = {
                "path": str(checkpoint_path),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        finally:
            bootstrap.close()
    contract = None
    for kind_index, kind in enumerate(KINDS):
        if kind in selected_conditions:
            collect_kind(args, kind, kind_index, rows, behavior)
    packed = {key: np.concatenate(value) for key, value in rows.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    # Infer the exact contract with a small bootstrap environment after data
    # collection, keeping names and stored width mechanically tied.
    env = make_env(1, "nominal", onset=0, force_scale=1.0, return_delay=30)
    try:
        obs, _ = env.reset(seed=args.seed_base - 1)
        contract = feature_contract(obs)
    finally:
        env.close()
    if len(contract) != packed["sequence"].shape[-1]:
        raise RuntimeError("feature-name contract does not match saved tensor")
    metadata = {
        "schema_version": 1, "conditions": list(KINDS),
        "collected_conditions": list(selected_conditions),
        "snapshots": [x for x in SNAPSHOTS if x <= args.horizon],
        "rows": int(len(packed["length"])),
        "simulator_batch_groups": int(len(np.unique(packed["group_id"]))),
        "feature_names": contract,
        "forbidden_feature_keys": list(FORBIDDEN_FEATURE_KEYS),
        "seed_base": args.seed_base,
        "onset_range": [args.onset_min, args.onset_max],
        "force_scale_range": [args.force_scale_min, args.force_scale_max],
        "return_delay_range": [args.return_delay_min, args.return_delay_max],
        "horizon": args.horizon,
        "collection_policy": (
            "zero action safe deferral" if behavior is None
            else f"closed-loop nominal visual policy index {args.policy_index}"
        ),
        "behavior_checkpoint": behavior_checkpoint,
        "prefix_timestamp": "pre_action_observation_matching_deployment",
        "split_unit": "entire vectorized simulator reset batch",
        "absolute_pose_features": False,
        "hand_engineered_temporal_features": False,
        "training_only_targets": ["mechanism family", "temporary_block_cleared"],
    }
    Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k != "feature_names"}, indent=2))


if __name__ == "__main__":
    main()
