"""Device selection for the torch-based perception models in this package
(clip_feasibility.py, dinov2_probe.py) -- CUDA when available, CPU
otherwise, so the same code runs unmodified on this CPU-only dev machine
today and a CUDA workstation/cluster later.

Deliberately NOT used for this package's ManiSkill env `sim_backend`
(unlike `maniskill_humanoid_spike/device_utils.py`'s `resolve_sim_backend()`,
which this package does not import): every env here
(tidy_up_env.py/tidy_up_env_humanoid.py/tidy_up_env_replicacad*.py) has a
hard, unconditional `RuntimeError` guard against GPU-batched sim, because
object add/remove -- the mechanism every intervention in this project uses
-- is unsupported under `physx_cuda` regardless of what hardware is
available. `sim_backend="physx_cpu"` there is a correctness requirement,
not a performance fallback, so auto-selecting CUDA when present would
break every episode rather than speed it up. See D-012 (the same
distinction, first found for the older maniskill_humanoid_spike) and each
env's own `_initialize_episode` guard.
"""

from __future__ import annotations

import torch


def resolve_torch_device(prefer_gpu: bool = True) -> torch.device:
    """CUDA when available and preferred, otherwise CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
