"""Training infrastructure shared by learned ATR policies."""

from atr.training.checkpointing import JsonCheckpointManager, TrainingCheckpoint

__all__ = ["JsonCheckpointManager", "TrainingCheckpoint"]
