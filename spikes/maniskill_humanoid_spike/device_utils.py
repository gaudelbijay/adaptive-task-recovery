"""Machine-agnostic simulation backend selection.

ManiSkill3's own `sim_backend="auto"` does NOT check CUDA availability — per
`mani_skill/envs/sapien_env.py`, it resolves purely on `num_envs` (>1 ->
"physx_cuda", else -> "physx_cpu"). That means a single-env run on a CUDA
machine would still land on CPU under "auto", and a multi-env run on a
CUDA-less machine would try "physx_cuda" and fail outright. This resolves
the backend from actual CUDA availability instead, so the same code runs
correctly on a CUDA workstation or a CPU-only dev machine (e.g. this one —
Apple Silicon macOS, no CUDA) without editing anything.
"""

from __future__ import annotations

import torch


def resolve_sim_backend(prefer_gpu: bool = True) -> str:
    """Return "physx_cuda" when available and preferred, otherwise "physx_cpu"."""
    if prefer_gpu and torch.cuda.is_available():
        return "physx_cuda"
    return "physx_cpu"


def gpu_sim_enabled(env) -> bool:
    """True if the given (unwrapped or wrapped) gym env is running GPU sim."""
    return env.unwrapped.scene.gpu_sim_enabled
