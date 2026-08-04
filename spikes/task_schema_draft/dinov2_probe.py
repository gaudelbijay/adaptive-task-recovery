"""Stage 4 of docs/00-project-overview.md's build-up order: swap in a
representation learned from unlabeled data, once stage 3 (any working
pretrained visual model) works at all.

Deliberately different from clip_feasibility.py's approach, not just a bigger model:
CLIP is trained with *language* supervision (image-text pairs) -- its
zero-shot judgment in clip_feasibility.py works by comparing an image against a
hand-written text prompt. DINOv2 (`facebookresearch/dinov2`, ViT-S/14) is
trained with **no labels or text at all** -- purely self-supervised on
images. It has no notion of "coffee can" or "existence." The only way to
use it for this task is to fit a small linear probe on a handful of labeled
(embedding, exists) examples and check whether the representation makes
that boundary linearly separable -- which is the standard way self-supervised
representations get evaluated, and is genuinely different from prompting.

Getting labeled examples here is unusually constrained, not incidental:
D-022 (ai-notes/decisions.md) is a confirmed, open, unfixed upstream
ManiSkill3 bug where this env's rendered frames desync from the actual
scene after roughly the second render-producing reset in one process.
clip_feasibility.py's tests stay inside that safe budget (2 renders, one process).
A probe needs more examples than that, so `collect_labeled_examples()`
below spawns a fresh subprocess per example (`capture_episode_subprocess.py`)
-- each one is always "the first" render-producing reset from the OS's
point of view, which is the only way to collect more than ~2 examples
without risking silently-corrupted training data.

Honesty about scale: D-021 pinned this env's scene layout for good reason
(G1's placement is only valid on one specific apartment layout). That
means every example collected here is visually almost the same scene --
the only real variation is which object is being asked about and whether
the scripted intervention has fired yet. This is not a meaningful test of
representation *generalization* (different objects, layouts, lighting) --
it's a much narrower question: does DINOv2's embedding of this exact crop
linearly separate "object present" from "object absent" at all, on the one
scene this project can currently render reliably. Treat accordingly.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from atr.feasibility.clip_feasibility import _OBJECT_VISUAL_CONFIG
import atr.envs.capture_episode_subprocess as _capture_module
from atr.device_utils import resolve_torch_device

# capture_episode_subprocess.py promoted to src/atr/envs/ (D-052) -- located
# via the module's own __file__ rather than a hardcoded relative path, so
# this doesn't break again if either file moves independently.
_CAPTURE_SCRIPT = Path(_capture_module.__file__)


@lru_cache(maxsize=1)
def _dinov2_model():
    import torch

    device = resolve_torch_device()
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    model.eval().to(device)
    return model, device


def dinov2_embed(crop: np.ndarray) -> np.ndarray:
    """Self-supervised embedding of an image crop -- no text, no labels.
    Returns the 384-dim CLS token from DINOv2 ViT-S/14."""
    import torch
    from PIL import Image

    model, device = _dinov2_model()
    # DINOv2 wants a multiple of the 14px patch size; 224 is its standard
    # pretraining resolution.
    img = Image.fromarray(crop).resize((224, 224))
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    with torch.no_grad():
        embedding = model(x)
    return embedding[0].cpu().numpy()


def collect_labeled_examples(
    object_id: str, n_present: int, n_absent: int, seed_start: int = 0,
    scene_variant: str = "kitchen_cabinet",
) -> list[tuple[np.ndarray, bool]]:
    """Spawns one subprocess per example (see module docstring for why) and
    returns [(crop, exists_label), ...]. `n_present` examples come from
    steps=0 (before the scripted intervention fires); `n_absent` from
    steps=6 (after it fires and only affects master_chef_can -- so this only
    makes sense called with object_id="master_chef_can" for the absent
    half; potted_meat_can never goes absent in this env's intervention).
    `scene_variant` (D-027): "kitchen_cabinet" (original) or "kitchen_sink"
    (second calibrated layout, added so this isn't validated on only one
    scene) -- see tidy_up_env_replicacad_humanoid.py's _SCENE_CONFIGS."""
    cfg = _OBJECT_VISUAL_CONFIG[scene_variant][object_id]
    y0, y1, x0, x1 = cfg.crop
    examples: list[tuple[np.ndarray, bool]] = []
    plan = [(0, n_present)] + ([(6, n_absent)] if n_absent > 0 else [])
    seed = seed_start
    for steps, count in plan:
        for _ in range(count):
            out_path = Path(f"/tmp/_repr_capture_{seed}.npz")
            subprocess.run(
                [
                    sys.executable, str(_CAPTURE_SCRIPT),
                    "--seed", str(seed), "--steps", str(steps), "--out", str(out_path),
                    "--scene-variant", scene_variant,
                ],
                check=True, capture_output=True,
            )
            data = np.load(out_path)
            frame = data["frame"]
            label = bool(data[f"exists_{object_id}"])
            examples.append((frame[y0:y1, x0:x1], label))
            out_path.unlink()
            seed += 1
    return examples


def fit_and_evaluate_probe(examples: list[tuple[np.ndarray, bool]]) -> dict:
    """Fits a logistic-regression linear probe on DINOv2 embeddings and
    evaluates it with leave-one-out cross-validation (the only sensible
    choice at this sample size -- see module docstring on scale)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut

    embeddings = np.stack([dinov2_embed(crop) for crop, _ in examples])
    labels = np.array([label for _, label in examples], dtype=int)

    predictions = []
    for train_idx, test_idx in LeaveOneOut().split(embeddings):
        probe = LogisticRegression(max_iter=1000).fit(embeddings[train_idx], labels[train_idx])
        predictions.append(probe.predict(embeddings[test_idx])[0])
    predictions = np.array(predictions)

    return {
        "accuracy": float((predictions == labels).mean()),
        "n_examples": len(examples),
        "n_positive": int(labels.sum()),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }
