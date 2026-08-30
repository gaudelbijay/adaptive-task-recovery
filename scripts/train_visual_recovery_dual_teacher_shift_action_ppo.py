#!/usr/bin/env python3
"""Run the frozen dual-teacher PPO loop with bounded action consistency."""

from __future__ import annotations

import hashlib
from pathlib import Path

import train_visual_recovery_dual_teacher_drac_ppo as trainer
from bounded_shift_action_consistency import bounded_shift_action_consistency


_SOURCE_PATHS = {
    # `trainer` is the generic checkpoint auditor's required canonical key.
    # Retain the explicit alias below so the wrapper/base lineage stays visible.
    "trainer": Path(__file__).resolve(),
    "trainer_wrapper": Path(__file__).resolve(),
    "base_trainer": Path(trainer.__file__).resolve(),
    "environment": Path(__file__).resolve().parents[1]
    / "src/atr/envs/learned_recovery.py",
    "environment_v3": Path(__file__).resolve().parents[1]
    / "src/atr/envs/learned_recovery_v3.py",
    "bounded_shift_action_consistency": Path(__file__).with_name(
        "bounded_shift_action_consistency.py"
    ),
}

# The base loop calls this symbol for its separately weighted consistency term.
# Override it before entering main and replace provenance with the actual method.
trainer.drac_policy_consistency = bounded_shift_action_consistency
trainer.SOURCE_SHA256 = {
    name: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, path in _SOURCE_PATHS.items()
}


if __name__ == "__main__":
    trainer.main()
