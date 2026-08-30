#!/usr/bin/env python3
"""Fail closed on incomplete, mismatched, or non-finite PPO checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def _tasks(config: dict) -> list[dict]:
    return [
        {**experiment, "seed": seed}
        for experiment in config["experiments"]
        for seed in config["seeds"]
    ]


def _finite_tensors(value, prefix="") -> tuple[int, list[str]]:
    count = 0
    failures = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            count = 1
            if not bool(torch.isfinite(value).all()):
                failures.append(prefix or "<root>")
    elif isinstance(value, dict):
        for key, item in value.items():
            child_count, child_failures = _finite_tensors(
                item, f"{prefix}.{key}" if prefix else str(key),
            )
            count += child_count
            failures.extend(child_failures)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child_count, child_failures = _finite_tensors(item, f"{prefix}[{index}]")
            count += child_count
            failures.extend(child_failures)
    return count, failures


def _finite_numbers(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(args.output) / config["name"]
    records = []
    for task in _tasks(config):
        method = task.get("method", task["env_id"])
        run_dir = root / method / f"seed_{int(task['seed'])}"
        best_path, latest_path = run_dir / "best.pt", run_dir / "latest.pt"
        if not best_path.exists() or not latest_path.exists():
            raise FileNotFoundError(f"checkpoint pair unavailable: {run_dir}")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        latest = torch.load(latest_path, map_location="cpu", weights_only=False)
        for label, checkpoint in (("best", best), ("latest", latest)):
            if checkpoint.get("task") != task:
                raise ValueError(f"{label} checkpoint task mismatch: {run_dir}")
            if not isinstance(checkpoint.get("agent"), dict):
                raise ValueError(f"{label} checkpoint lacks agent state: {run_dir}")
            tensor_count, failures = _finite_tensors(checkpoint["agent"], "agent")
            if tensor_count == 0 or failures:
                raise ValueError(
                    f"{label} checkpoint has invalid agent tensors: {failures}"
                )
            if not _finite_numbers(checkpoint.get("best_metrics", {})):
                raise ValueError(f"{label} checkpoint has non-finite metrics")
        optimizer_count, optimizer_failures = _finite_tensors(
            latest.get("optimizer", {}), "optimizer",
        )
        if optimizer_count == 0 or optimizer_failures:
            raise ValueError(
                f"latest checkpoint has invalid optimizer tensors: {optimizer_failures}"
            )
        batch = int(task["num_envs"]) * int(task["num_steps"])
        scheduled = int(task["total_timesteps"]) // batch * batch
        latest_step = int(latest.get("global_step", -1))
        best_step = int(best.get("global_step", -1))
        if latest_step != scheduled:
            raise ValueError(
                f"latest checkpoint budget mismatch: {latest_step} != {scheduled}"
            )
        if not 0 <= best_step <= latest_step:
            raise ValueError("best checkpoint step is outside the completed budget")
        kind = "visual" if "observation_contract" in best else "state"
        if kind == "visual":
            if best.get("observation_contract") != latest.get("observation_contract"):
                raise ValueError("visual checkpoint observation contracts disagree")
            for label, checkpoint in (("best", best), ("latest", latest)):
                source = checkpoint.get("source_sha256")
                if not isinstance(source, dict) or not source.get("trainer") or not source.get("environment"):
                    raise ValueError(f"{label} visual checkpoint lacks source provenance")
        records.append({
            "method": method,
            "training_seed": int(task["seed"]),
            "kind": kind,
            "scheduled_global_step": scheduled,
            "best_global_step": best_step,
            "latest_global_step": latest_step,
            "best_iteration": int(best["iteration"]),
            "latest_iteration": int(latest["iteration"]),
            "agent_floating_tensor_count": _finite_tensors(best["agent"])[0],
            "optimizer_floating_tensor_count": optimizer_count,
            "all_finite": True,
            "best_checkpoint_sha256": _sha256(best_path),
            "latest_checkpoint_sha256": _sha256(latest_path),
            "observation_contract": best.get("observation_contract"),
            "training_source_sha256": best.get("source_sha256"),
        })
    payload = {
        "schema_version": 1,
        "protocol": "post-training immutable checkpoint integrity audit",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "experiment": config["name"],
        "expected_tasks": len(_tasks(config)),
        "audited_tasks": len(records),
        "all_complete_and_finite": True,
        "records": records,
        "auditor_sha256": _sha256(Path(__file__)),
    }
    target = Path(args.audit_output) if args.audit_output else root / "checkpoint_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
