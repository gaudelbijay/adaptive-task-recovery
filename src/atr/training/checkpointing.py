"""Atomic, configuration-safe checkpoints for interruptible training jobs.

Jarvis limits an uninterrupted job to 24 hours.  A useful checkpoint must
therefore preserve optimizer/learner state *and* the random-number state: using
the same seed while restarting from episode zero is not a resume.  This module
keeps the storage format JSON so checkpoints remain inspectable and avoids
loading arbitrary pickle payloads.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingCheckpoint:
    schema_version: int
    config_fingerprint: str
    completed_episodes: int
    learner_state: dict[str, Any]
    rng_state_repr: str
    validation_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "completed_episodes": self.completed_episodes,
            "learner_state": self.learner_state,
            "rng_state_repr": self.rng_state_repr,
            "validation_score": self.validation_score,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingCheckpoint":
        required = {
            "schema_version", "config_fingerprint", "completed_episodes",
            "learner_state", "rng_state_repr", "validation_score",
        }
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"checkpoint missing fields: {sorted(missing)}")
        if payload["schema_version"] != 1:
            raise ValueError(f"unsupported checkpoint schema {payload['schema_version']!r}")
        return cls(
            schema_version=1,
            config_fingerprint=str(payload["config_fingerprint"]),
            completed_episodes=int(payload["completed_episodes"]),
            learner_state=dict(payload["learner_state"]),
            rng_state_repr=str(payload["rng_state_repr"]),
            validation_score=(
                None if payload["validation_score"] is None
                else float(payload["validation_score"])
            ),
        )

    def rng_state(self) -> tuple:
        """Return the exact ``random.Random`` state using safe parsing."""
        state = ast.literal_eval(self.rng_state_repr)
        if not isinstance(state, tuple):
            raise ValueError("checkpoint RNG state is not a tuple")
        return state


class JsonCheckpointManager:
    """Read/write ``latest.json`` and validation-selected ``best.json``."""

    def __init__(self, directory: str | Path, config_fingerprint: str):
        self.directory = Path(directory)
        self.config_fingerprint = config_fingerprint
        self.latest_path = self.directory / "latest.json"
        self.best_path = self.directory / "best.json"

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save(self, checkpoint: TrainingCheckpoint) -> bool:
        if checkpoint.config_fingerprint != self.config_fingerprint:
            raise ValueError("refusing to save checkpoint for a different configuration")
        self._atomic_write(self.latest_path, checkpoint.to_dict())

        previous_best = self.load_path(self.best_path, required=False)
        is_best = (
            checkpoint.validation_score is not None
            and (
                previous_best is None
                or previous_best.validation_score is None
                or checkpoint.validation_score > previous_best.validation_score
            )
        )
        if is_best:
            self._atomic_write(self.best_path, checkpoint.to_dict())
        return is_best

    def load_path(self, path: Path, *, required: bool = True) -> TrainingCheckpoint | None:
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        checkpoint = TrainingCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if checkpoint.config_fingerprint != self.config_fingerprint:
            raise ValueError(
                "checkpoint configuration mismatch: "
                f"expected {self.config_fingerprint}, found {checkpoint.config_fingerprint}"
            )
        return checkpoint

    def load_latest(self, *, required: bool = False) -> TrainingCheckpoint | None:
        return self.load_path(self.latest_path, required=required)

    def load_best(self, *, required: bool = False) -> TrainingCheckpoint | None:
        return self.load_path(self.best_path, required=required)
