"""Stage 3 of docs/00-project-overview.md's build-up order: replace the
privileged-state oracle with a feasibility judgment from images, starting
with any working pretrained visual model. Zero-shot CLIP (open_clip,
ViT-B-32, OpenAI weights) — no training, no fine-tuning.

Four real findings from getting this to work at all (measured, not assumed):

1. **Whole-frame zero-shot CLIP does not work here.** A global image/text
   similarity score for "a photo of a blue bowl" barely moved when the bowl
   was actually removed from the scene (measured delta ~0.01, sometimes
   the wrong sign, across 20 seeds on tidy_up_env.py). The object is a small
   fraction of a cluttered frame, which is a known CLIP weak point. A tight
   crop around the object's known on-screen location works far better --
   this is camera calibration (a fixed crop for a fixed camera pose), not a
   read of live 3D object position, so it doesn't leak privileged state.
2. **tidy_up_env.py's objects are plain colored boxes, not the objects
   they're named after** (`build_box` primitives standing in for "mug" /
   "bowl" -- see that file's docstring). Zero-shot CLIP correctly can't
   recognize "a blue bowl" in a picture of a blue cube, because there isn't
   one. This isn't a CLIP failure, it's a scene-realism mismatch. This
   module is calibrated against `tidy_up_env_replicacad_humanoid.py`
   instead, which places real, photorealistic YCB-scanned objects (D-017)
   in frame of a fixed camera (D-018) -- a fair test of whether zero-shot
   vision can do this at all.

3. **`_trigger_intervention()` removed the object from physics without
   refreshing the render scene graph** (`tidy_up_env_replicacad_humanoid.py`)
   -- every existing consumer of this env reads privileged state, not
   pixels, so a stale render went unnoticed until this was the first thing
   to actually look at a frame after a removal. Fixed by adding
   `self.scene.update_render()` to that branch, matching the pattern the
   `temporary_obstacle` branch already used.
4. **G1's hardcoded base pose and camera were originally calibrated for
   exactly one apartment layout.** `ReplicaCADSetTableTrain` loads a
   different room per seed -- rendering seed=2 for this check landed G1
   next to a couch and a bicycle, nowhere near the cans. Every prior test
   of this env (D-018) only ever used seed=0, so this went unnoticed until
   vision work rendered and looked at other seeds. Fixed at the
   scene-layout level in D-021 (pinning, not sampling); a *second*
   calibrated layout ("kitchen_sink") was added in D-027 specifically so
   this module's crop/prompt calibration isn't validated on only one scene
   -- see `_OBJECT_VISUAL_CONFIG` below, now keyed per scene variant.

Generic object descriptions ("a photo of a green can") also underperformed
specific/iconic ones ("a photo of a Spam can") by a wide margin -- CLIP's
web-scale training data apparently has much stronger associations for
recognizable brand-name objects than generic category descriptions. This is
why `_OBJECT_VISUAL_CONFIG` below has a hand-picked crop and
prompt per object instead of a generic "a photo of a {object_id}" template
-- that generic template was tried first and measured to perform worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class VisualObjectConfig:
    """Where to look in the (fixed) camera frame, and what to ask CLIP."""

    crop: tuple[int, int, int, int]  # (y0, y1, x0, x1)
    positive_prompt: str
    negative_prompt: str = "a photo of an empty cabinet with nothing on it"


# Calibrated per scene variant (tidy_up_env_replicacad_humanoid.py's
# _SCENE_CONFIGS) at 512x512, each against that variant's own fixed camera.
# Crop pixels found by saving a frame and either inspecting it directly or
# (kitchen_sink) projecting the object's known world position through the
# camera's own intrinsic/extrinsic matrices and cropping around that pixel
# -- not a read of live 3D position at call time, so it doesn't leak
# privileged state; it's camera calibration, done once, same as
# kitchen_cabinet's crops were (found by inspection instead, since that
# camera's framing was simple enough not to need projection math).
_OBJECT_VISUAL_CONFIG: dict[str, dict[str, VisualObjectConfig]] = {
    "kitchen_cabinet": {
        "master_chef_can": VisualObjectConfig(
            crop=(180, 380, 260, 460), positive_prompt="a photo of a coffee can",
        ),
        "potted_meat_can": VisualObjectConfig(
            crop=(180, 380, 60, 260), positive_prompt="a photo of a Spam can",
        ),
    },
    # D-027: master_chef_can sits in the open on a counter here; potted_meat_can
    # sits inside a sink basin, which is why its crop/prompt needed a
    # steeper camera angle and different negative prompt to separate at all
    # -- see ai-notes/decisions.md D-027 for the calibration process.
    "kitchen_sink": {
        "master_chef_can": VisualObjectConfig(
            crop=(265, 365, 305, 405), positive_prompt="a photo of a blue can",
            negative_prompt="a photo of an empty countertop",
        ),
        "potted_meat_can": VisualObjectConfig(
            crop=(139, 239, 146, 246), positive_prompt="a photo of a Spam can",
            negative_prompt="a photo of an empty sink",
        ),
    },
}


@lru_cache(maxsize=1)
def _clip_model():
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    model.eval()
    return model, preprocess, tokenizer


def clip_margin(image: np.ndarray, positive_prompt: str, negative_prompt: str) -> float:
    """cosine_similarity(image, positive) - cosine_similarity(image, negative).
    Positive means the image looks more like `positive_prompt`."""
    import torch
    from PIL import Image

    model, preprocess, tokenizer = _clip_model()
    img = preprocess(Image.fromarray(image)).unsqueeze(0)
    tok = tokenizer([positive_prompt, negative_prompt])
    with torch.no_grad():
        img_f = model.encode_image(img)
        txt_f = model.encode_text(tok)
        img_f /= img_f.norm(dim=-1, keepdim=True)
        txt_f /= txt_f.norm(dim=-1, keepdim=True)
        sims = (img_f @ txt_f.T)[0].tolist()
    return sims[0] - sims[1]


def visual_object_exists(
    frame: np.ndarray, object_id: str, scene_variant: str = "kitchen_cabinet",
) -> bool:
    """Zero-shot judgment of whether `object_id` is visible in `frame`,
    using the calibrated crop/prompt in _OBJECT_VISUAL_CONFIG[scene_variant].
    Raises for objects/variants that don't have a calibrated config -- there
    is no generic fallback, per this module's docstring finding that the
    generic prompt template measurably doesn't work."""
    if scene_variant not in _OBJECT_VISUAL_CONFIG:
        raise ValueError(
            f"no calibrated visual config for scene_variant={scene_variant!r}; "
            f"known variants: {sorted(_OBJECT_VISUAL_CONFIG)}"
        )
    variant_config = _OBJECT_VISUAL_CONFIG[scene_variant]
    if object_id not in variant_config:
        raise ValueError(
            f"no calibrated visual config for {object_id!r} in scene_variant={scene_variant!r}; "
            f"known objects: {sorted(variant_config)}"
        )
    cfg = variant_config[object_id]
    y0, y1, x0, x1 = cfg.crop
    crop = frame[y0:y1, x0:x1]
    margin = clip_margin(crop, cfg.positive_prompt, cfg.negative_prompt)
    return margin > 0.0
