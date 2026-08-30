"""Task-reward-only visual encoder (D-066) -- the baseline H1
(docs/01-problem-statement-and-motivation.md) is actually about: "self-
supervised visual representations improve feasibility prediction... over
pixels trained only through task reward and standard supervised
features." Nothing in this project had built that comparison point
before this. Both existing perceptual models start from a large
pretrained backbone: CLIP (`clip_feasibility.py`, D-020) is language-
supervised (image-text alignment on web-scale data), DINOv2
(`dinov2_probe.py`, D-023) is self-supervised (ImageNet-scale, no
labels) -- neither is "pixels trained only through task reward." This
module is: a small encoder, randomly initialized, no pretrained weights
of any kind, trained end-to-end from scratch on nothing but this
project's own tiny labeled dataset and its own reward shape.

Honesty about the simplification: this trains via a reward-*derived*
supervised loss (binary cross-entropy against the reward-optimal
action), not literal online policy-gradient RL rolling out actions in
the environment. For this project's specific decision (attempt iff the
object exists), "attempt iff exists" is also exactly the reward-optimal
action under `q_learning.py`'s own reward shape (+1.0 achieved,
`-0.1 * steps_used` otherwise) -- so the existence label already used to
train DINOv2's probe doubles as the reward-optimal action label here.
What's genuinely different is not the label source, it's the *learning
signal behind the visual features themselves*: no self-supervised or
language-supervised pretraining anywhere in this pipeline, trained from
scratch on this task's own ~24 examples, the same toy sample size CLIP
and DINOv2 were evaluated against. A literal from-environment policy-
gradient version is a real, larger future step, not attempted here --
this is the direct, honest comparison point H1 asks for at this
project's current scale, not a claim to have run full RL-from-pixels.

**Measured result and correction (D-066, 2026-08-28): the encoder does
not generalize from this toy-scale data.** The original captures gave 0%
leave-one-out (LOO) accuracy and fold-wise majority-class outputs.  A fresh
Jarvis capture still gave chance-or-worse LOO accuracy, while a fit on all 12
balanced examples separated its six positive and six negative training images
strongly (logits about +4.18 and -4.07).  Thus the supported result is failed
held-out generalization at n=12, not a platform-independent claim that the
optimizer or representation always collapses to a constant.  Gradient flow
and weight changes are separately tested.  CLIP and DINOv2 remain stronger on
this same narrow LOO comparison, but all conclusions remain bounded to one
object/scene and a reward-derived supervised proxy rather than online RL.
"""

from __future__ import annotations

import numpy as np


def _to_tensor(crop: np.ndarray):
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.array(crop)).float().permute(2, 0, 1) / 255.0
    x = F.interpolate(x.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def _build_model(seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=5, stride=2), nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=5, stride=2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1),
    )


def train_task_reward_encoder(
    examples: list[tuple[np.ndarray, bool]], epochs: int = 300, lr: float = 0.01, seed: int = 0,
):
    """Trains a tiny, randomly-initialized conv encoder (3 conv/pool
    layers + a linear head -- deliberately small, matching this project's
    toy-scale data; a bigger network would just memorize ~24 examples)
    end-to-end via `BCEWithLogitsLoss` against the reward-optimal action
    label. Returns the trained `torch.nn.Module`; predict with
    `predict_exists()` below."""
    import torch
    import torch.nn as nn

    model = _build_model(seed)
    crops = torch.stack([_to_tensor(crop) for crop, _ in examples])
    labels = torch.tensor([float(label) for _, label in examples])

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(crops).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_logit(model, crop: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        return model(_to_tensor(crop).unsqueeze(0)).squeeze(-1).item()


def predict_exists(model, crop: np.ndarray) -> bool:
    return predict_logit(model, crop) > 0


def fit_and_evaluate_encoder(
    examples: list[tuple[np.ndarray, bool]], epochs: int = 300, lr: float = 0.01, seed: int = 0,
) -> dict:
    """Leave-one-out cross-validation, the exact same procedure and
    sample size `dinov2_probe.fit_and_evaluate_probe()` uses -- same
    metric, same evaluation discipline, so the two are directly
    comparable. Each fold trains a fresh encoder from scratch (excluding
    the held-out example), the honest LOO discipline this project already
    established for DINOv2, not a cheaper approximation."""
    predictions = []
    labels = [label for _, label in examples]
    for i in range(len(examples)):
        train_examples = examples[:i] + examples[i + 1:]
        held_out_crop, held_out_label = examples[i]
        model = train_task_reward_encoder(train_examples, epochs=epochs, lr=lr, seed=seed)
        predictions.append(predict_exists(model, held_out_crop))

    predictions = np.array(predictions, dtype=int)
    labels = np.array(labels, dtype=int)
    return {
        "accuracy": float((predictions == labels).mean()),
        "n_examples": len(examples),
        "n_positive": int(labels.sum()),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }
