#!/usr/bin/env python3
"""Train pinned NE-Dreamer from RGB and robot proprioception on LearnedRecovery.

The adapter deliberately excludes object/goal/sweeper poses, TCP pose, task
progress, and oracle unavailability.  The policy receives one RGB camera,
joint position/velocity, and the parsed two-token instruction.  Environment
reward and success computation may use simulator state, as is standard for
vision-based RL, but that state never enters the observation encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


STOP_REQUESTED = False


def _request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"checkpoint requested by signal {signum}", flush=True)


def _atomic_torch_save(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _config_fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ManiSkillVisionBatch:
    """In-process GPU-vectorized ManiSkill adapter for NE-Dreamer."""

    def __init__(self, task: dict, env_num: int, seed: int, evaluation: bool):
        import atr.envs.learned_recovery  # noqa: F401

        self.env_num = int(env_num)
        self.device = torch.device("cuda:0")
        self.episode_steps = int(task["episode_steps"])
        kwargs = dict(task["eval_env_kwargs"] if evaluation else task["env_kwargs"])
        kwargs["vision_camera_size"] = int(task["camera_size"])
        self.env = gym.make(
            "LearnedRecovery-v1",
            num_envs=self.env_num,
            obs_mode="rgb",
            render_mode=None,
            sim_backend="physx_cuda",
            control_mode=task["control_mode"],
            reward_mode=task["reward_mode"],
            reconfiguration_freq=1,
            **kwargs,
        )
        self._seed = int(seed)
        self._needs_reset = True
        self._episode_generation = 0
        raw_action_space = self.env.action_space
        action_dim = int(raw_action_space.shape[-1])
        low = np.asarray(raw_action_space.low).reshape(-1, action_dim)[0]
        high = np.asarray(raw_action_space.high).reshape(-1, action_dim)[0]
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, (task["camera_size"], task["camera_size"], 3), np.uint8),
            "proprio": gym.spaces.Box(-np.inf, np.inf, (18,), np.float32),
            "instruction": gym.spaces.Box(0.0, 1.0, (2,), np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), bool),
            "is_last": gym.spaces.Box(0, 1, (1,), bool),
            "is_terminal": gym.spaces.Box(0, 1, (1,), bool),
            "success": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "constraint_violated": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        })

    def _transition(self, obs, reward, first, last, info=None):
        info = info or {}
        image = obs["sensor_data"]["base_camera"]["rgb"]
        proprio = torch.cat([obs["agent"]["qpos"], obs["agent"]["qvel"]], dim=-1).float()
        instruction = obs["extra"]["instruction"].float()
        zeros = torch.zeros(self.env_num, device=self.device)
        success = info.get("success", zeros).float()
        violation = info.get("constraint_violated", zeros).float()
        data = {
            "image": image,
            "proprio": proprio,
            "instruction": instruction,
            "is_first": torch.full((self.env_num, 1), first, device=self.device, dtype=torch.bool),
            "is_last": torch.full((self.env_num, 1), last, device=self.device, dtype=torch.bool),
            "is_terminal": torch.zeros((self.env_num, 1), device=self.device, dtype=torch.bool),
            "success": success.reshape(self.env_num, 1),
            "constraint_violated": violation.reshape(self.env_num, 1),
            "reward": reward.reshape(self.env_num, 1).float(),
        }
        return TensorDict(data, batch_size=(self.env_num,), device=self.device)

    def step(self, action, done):
        if self._needs_reset or bool(torch.as_tensor(done).all()):
            self._episode_generation += 1
            obs, _ = self.env.reset(seed=self._seed + self._episode_generation)
            self._needs_reset = False
            reward = torch.zeros(self.env_num, device=self.device)
            trans = self._transition(obs, reward, first=True, last=False)
            return trans, torch.zeros(self.env_num, device=self.device, dtype=torch.bool)

        action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if bool((torch.abs(action) > 1.000001).any()):
            maximum = float(torch.abs(action).max())
            raise RuntimeError(f"unbounded policy action reached ManiSkill: max_abs={maximum}")
        obs, reward, _terminated, truncated, info = self.env.step(action)
        # LearnedRecovery has a common fixed time limit.  We deliberately train
        # through early success/failure signals, matching PPO's
        # ignore_terminations protocol and avoiding asynchronous reset leakage.
        done_out = truncated.bool()
        last = bool(done_out.all())
        if bool(done_out.any()) and not last:
            raise RuntimeError("vision batch expected synchronous time-limit truncation")
        self._needs_reset = last
        return self._transition(obs, reward, first=False, last=last, info=info), done_out

    def close(self):
        self.env.close()


class GracefulStop(Exception):
    pass


class _StraightThroughBound:
    """Elementwise action bound with the upstream straight-through gradient.

    NE-Dreamer's ``Bound`` implements ``sample`` but Dreamer calls ``rsample``;
    attribute forwarding therefore bypasses the bound.  This implementation
    covers both sampling APIs and is used for environment and imagination
    actions, ensuring replay contains the command the controller executed.
    """

    def __init__(self, distribution):
        self._distribution = distribution

    def __getattr__(self, name):
        return getattr(self._distribution, name)

    @staticmethod
    def _bound(value):
        return value / torch.clamp(torch.abs(value), min=1.0).detach()

    @property
    def mode(self):
        return self._bound(self._distribution.mean)

    def rsample(self, sample_shape=torch.Size()):
        return self._bound(self._distribution.rsample(sample_shape))

    def sample(self, sample_shape=torch.Size()):
        return self._bound(self._distribution.sample(sample_shape))

    def log_prob(self, value):
        return self._distribution.log_prob(value)

    def entropy(self):
        return self._distribution.entropy()


def _install_bounded_action_distribution(distribution_module):
    original = distribution_module.bounded_normal

    def bounded_normal(*args, **kwargs):
        return _StraightThroughBound(original(*args, **kwargs))

    distribution_module.bounded_normal = bounded_normal


def _read_last_eval(metrics_path: Path, step: int) -> dict:
    if not metrics_path.exists():
        return {}
    for line in reversed(metrics_path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        if int(record.get("step", -1)) == int(step) and "episode/eval_score" in record:
            return record
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learned_recovery_nedreamer_pilot.json")
    parser.add_argument("--output", default="results/vision_nedreamer")
    parser.add_argument("--task-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    task = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = list(task["seeds"])
    if not 0 <= args.task_index < len(seeds):
        raise ValueError(f"task-index must be in [0, {len(seeds) - 1}]")
    seed = int(seeds[args.task_index])
    upstream = Path(args.upstream).resolve()
    # Jarvis GPU images do not expose the Git executable.  The checkout is
    # deliberately detached, so its HEAD file is the exact immutable commit.
    commit = (upstream / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if commit != task["upstream_commit"]:
        raise RuntimeError(f"NE-Dreamer checkout mismatch: {commit}")
    if args.preflight:
        print(json.dumps({"seed": seed, "task_count": len(seeds), "upstream": commit}, indent=2))
        return

    sys.path.insert(0, str(upstream))
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(upstream / "configs")):
        config = compose(config_name="configs", overrides=[
            f"model={task['model_size']}",
            f"model.rep_loss={task['algorithm']}",
            "model.compile=false",
            "model.imagination_decoding.enabled=false",
            "model.posthoc_decoder.enabled=false",
            "model.saliency.enabled=false",
            "device=cuda:0",
            f"seed={seed}",
            f"batch_size={task['batch_size']}",
            f"batch_length={task['batch_length']}",
            f"buffer.max_size={task['replay_capacity']}",
            f"trainer.steps={task['total_environment_steps']}",
            f"trainer.eval_every={task['eval_every']}",
            f"trainer.eval_episode_num={task['num_eval_envs']}",
            f"trainer.train_ratio={task['train_ratio']}",
            f"trainer.update_log_every={task['log_every']}",
            "trainer.eval_video_every=0",
            "trainer.s3_bucket=null",
            "env.action_repeat=1",
            f"env.time_limit={task['episode_steps']}",
            "env.encoder.cnn_keys=image",
            "env.encoder.mlp_keys=proprio|instruction",
            "env.decoder.cnn_keys=image",
            "env.decoder.mlp_keys=proprio|instruction",
        ])

    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_float32_matmul_precision("high")

    distribution_mod = importlib.import_module("distributions")
    _install_bounded_action_distribution(distribution_mod)
    dreamer_mod = importlib.import_module("dreamer")
    trainer_mod = importlib.import_module("trainer")
    buffer_mod = importlib.import_module("buffer")
    tools = importlib.import_module("tools")

    run_dir = Path(args.output) / task["name"] / task["algorithm"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _config_fingerprint(task)
    (run_dir / "task.json").write_text(
        json.dumps({**task, "selected_seed": seed, "config_sha256": fingerprint}, indent=2) + "\n",
        encoding="utf-8",
    )
    config.logdir = str(run_dir)
    logger = tools.Logger(run_dir)
    logger.log_hydra_config(config)
    replay = buffer_mod.Buffer(config.buffer)
    train_env = ManiSkillVisionBatch(task, task["num_envs"], seed, evaluation=False)
    eval_episodes = int(args.eval_episodes or task["num_eval_envs"])
    eval_env = ManiSkillVisionBatch(task, eval_episodes, seed + 100000, evaluation=True)
    agent = dreamer_mod.Dreamer(config.model, train_env.observation_space, train_env.action_space).to(config.device)

    latest = run_dir / "latest.pt"
    best = run_dir / "best.pt"
    best_metric_path = run_dir / "best_metric.json"
    replay_dir = run_dir / "replay_latest"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=config.device, weights_only=False)
        if checkpoint["config_sha256"] != fingerprint:
            raise RuntimeError("checkpoint does not match immutable experiment config")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        agent._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if replay_dir.exists() and not args.eval_only:
            replay._buffer.loads(replay_dir)
        print(f"resumed checkpoint at {checkpoint['environment_steps']} environment steps", flush=True)

    class CheckpointingTrainer(trainer_mod.OnlineTrainer):
        def save_checkpoint(self, reason: str, step: int | None = None):
            environment_steps = int(replay.count() * config.env.action_repeat if step is None else step)
            evaluation = _read_last_eval(run_dir / "metrics.jsonl", environment_steps)
            payload = {
                "schema_version": 1,
                "algorithm": task["algorithm"],
                "upstream_commit": task["upstream_commit"],
                "config_sha256": fingerprint,
                "environment_steps": environment_steps,
                "saved_at_unix": time.time(),
                "reason": reason,
                "evaluation": evaluation,
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "scheduler_state_dict": agent._scheduler.state_dict(),
                "scaler_state_dict": agent._scaler.state_dict(),
            }
            _atomic_torch_save(payload, latest)
            if evaluation:
                metric = [
                    float(evaluation["episode/eval_success_rate"]),
                    float(evaluation["episode/eval_score"]),
                ]
                previous = (
                    json.loads(best_metric_path.read_text(encoding="utf-8"))["metric"]
                    if best_metric_path.exists() else [-float("inf"), -float("inf")]
                )
                if metric > previous:
                    _atomic_torch_save(payload, best)
                    best_metric_path.write_text(json.dumps({
                        "metric": metric,
                        "environment_steps": environment_steps,
                        "selection": "lexicographic_success_then_return",
                    }, indent=2) + "\n", encoding="utf-8")
            if replay.count():
                temporary = run_dir / f".replay.tmp.{os.getpid()}"
                if temporary.exists():
                    shutil.rmtree(temporary)
                replay._buffer.dumps(temporary)
                old = run_dir / ".replay.previous"
                if old.exists():
                    shutil.rmtree(old)
                if replay_dir.exists():
                    os.replace(replay_dir, old)
                os.replace(temporary, replay_dir)
                if old.exists():
                    shutil.rmtree(old)
            print(f"checkpoint saved: reason={reason} steps={environment_steps}", flush=True)

        def eval(self, selected_agent, train_step):
            super().eval(selected_agent, train_step)
            self.save_checkpoint("evaluation", int(train_step))

    trainer = CheckpointingTrainer(
        config.trainer, replay, logger, run_dir, train_env, eval_env, agent.act_dim
    )
    trainer.eval_episode_num = eval_episodes
    if args.eval_only:
        if not latest.exists():
            raise RuntimeError("--eval-only requires an existing latest.pt checkpoint")
        environment_steps = int(checkpoint["environment_steps"])
        trainer.eval(agent, environment_steps)
        (run_dir / f"EVALUATION_{eval_episodes}_COMPLETE.json").write_text(
            json.dumps({
                "environment_steps": environment_steps,
                "episodes": eval_episodes,
                "completed_at_unix": time.time(),
            }) + "\n",
            encoding="utf-8",
        )
        train_env.close()
        eval_env.close()
        return

    original_update = agent.update

    def checked_update(selected_replay):
        metrics = original_update(selected_replay)
        if STOP_REQUESTED:
            trainer.save_checkpoint("slurm_signal")
            raise GracefulStop
        return metrics

    agent.update = checked_update
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        trainer.begin(agent)
    except GracefulStop:
        print("graceful checkpoint boundary reached", flush=True)
    else:
        trainer.save_checkpoint("complete", int(task["total_environment_steps"]))
        (run_dir / "TRAINING_COMPLETE.json").write_text(
            json.dumps({"environment_steps": task["total_environment_steps"], "completed_at_unix": time.time()}) + "\n",
            encoding="utf-8",
        )
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
