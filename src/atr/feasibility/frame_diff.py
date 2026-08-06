"""Frame-difference change detector -- the "simple pixel-difference change
detector plus rules" docs/10-evaluation-and-benchmarks.md's required-
baselines list asks for explicitly (and docs/08-training-pipeline.md's
stage 3 gate: "beat simple pixel-difference... baselines on held-out
compositions"), which had no first instance until this (D-063). Neither
CLIP (`clip_feasibility.py`, D-020, language-supervised) nor DINOv2
(`dinov2_probe.py`, D-023, self-supervised) is this baseline -- both learn
*something* (a text-image alignment, a fitted probe). This detector learns
nothing at all: no parameters, no supervision, no training data. It just
measures how much the pixels in a calibrated crop changed between two
frames and applies a fixed threshold ("changed a lot" -> "treat the goal
as no longer feasible"). Exists to test whether CLIP/DINOv2's added
complexity earns its keep at this project's current toy scale, or whether
a much dumber detector gets the same result here.

Deliberately reuses the same calibrated crop regions CLIP/DINOv2 already
use (`clip_feasibility._OBJECT_VISUAL_CONFIG`) rather than defining a new
set -- same reasoning `dinov2_probe.py` already relies on for that shared
config, and the fairest possible comparison: same crop, three different
judgments.
"""

from __future__ import annotations

import numpy as np


def frame_difference_score(before: np.ndarray, after: np.ndarray) -> float:
    """Mean absolute pixel difference (0-255 scale, matching the uint8
    frames every capture in this project already produces) between two
    same-shaped crops. The simplest possible pixel-difference signal --
    no learned parameters, no supervision of any kind."""
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch: before={before.shape} after={after.shape}")
    return float(np.abs(before.astype(np.float32) - after.astype(np.float32)).mean())


def object_changed(before: np.ndarray, after: np.ndarray, threshold: float) -> bool:
    """The "plus rules" half of the required baseline: a fixed threshold
    on `frame_difference_score()`, not a learned decision boundary.
    Callers combine this with the same rule CLIP/DINOv2 features feed
    into elsewhere in this project (e.g. `goal_feasible()`-style
    attempt/skip logic) -- this function only answers "did the pixels
    change," not "is the goal feasible," which is a policy-level
    decision, not a perception-level one."""
    return frame_difference_score(before, after) > threshold
