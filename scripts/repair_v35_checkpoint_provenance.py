#!/usr/bin/env python3
"""Add the generic environment provenance alias to completed V35 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch

from train_visual_recovery_dual_teacher_ppo import atomic_save, file_sha256, select_task


def tensor_digest(mapping: dict) -> str:
    digest = hashlib.sha256()
    tensors = []

    def visit(value, path):
        if torch.is_tensor(value):
            tensors.append((path, value))
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda item: str(item)):
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")

    visit(mapping, "root")
    for path, value in tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(path.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", default=Path("results/visual_recovery_ppo"), type=Path)
    parser.add_argument("--task-index", default=0, type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    task, _ = select_task(config, args.task_index)
    run_dir = args.output / config["name"] / task["method"] / f"seed_{task['seed']}"
    registration = importlib.import_module(task["registration_module"])
    environment_hash = hashlib.sha256(Path(registration.__file__).read_bytes()).hexdigest()
    report = {
        "schema_version": 1,
        "protocol": "V35 post-training provenance-key compatibility repair",
        "repair_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment_sha256": environment_hash,
        "artifacts": [],
        "claim_boundary": (
            "Metadata-only repair before evaluation. Agent and optimizer tensor "
            "digests must remain byte-identical; no metric or weight is changed."
        ),
    }
    for name in ("best.pt", "latest.pt"):
        path = run_dir / name
        before_file = file_sha256(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_protocol") != "supervised_translation_repair_v34":
            raise ValueError("unexpected V35 training protocol")
        source = checkpoint.get("source_sha256")
        if source.get("multicamera_environment") != environment_hash:
            raise ValueError("recorded multicamera environment hash mismatch")
        agent_before = tensor_digest(checkpoint["agent"])
        optimizer_before = tensor_digest(checkpoint["optimizer"])
        if source.get("environment") not in (None, environment_hash):
            raise ValueError("conflicting generic environment provenance")
        source["environment"] = environment_hash
        atomic_save(checkpoint, path)
        repaired = torch.load(path, map_location="cpu", weights_only=False)
        agent_after = tensor_digest(repaired["agent"])
        optimizer_after = tensor_digest(repaired["optimizer"])
        if agent_after != agent_before or optimizer_after != optimizer_before:
            raise RuntimeError("metadata repair changed checkpoint tensors")
        report["artifacts"].append({
            "path": str(path), "before_sha256": before_file,
            "after_sha256": file_sha256(path),
            "agent_tensor_sha256": agent_after,
            "optimizer_tensor_sha256": optimizer_after,
        })
    completion_path = run_dir / "TRAINING_COMPLETE.json"
    completion = json.loads(completion_path.read_text())
    completion["source_sha256"]["environment"] = environment_hash
    temporary = completion_path.with_name(f".{completion_path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, completion_path)
    report_path = run_dir / "PROVENANCE_REPAIR.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
