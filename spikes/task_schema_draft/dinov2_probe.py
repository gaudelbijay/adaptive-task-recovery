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
linearly separate "object present" from "object absent" at all. Tested on
two scene layouts as of D-053 ("kitchen_cabinet" and "kitchen_sink",
D-027), matching CLIP's 2-scene validation, not just one -- both 100%
leave-one-out accuracy.

Wired into a real live decision loop as of D-054
(`run_end_to_end_episode_dinov2()` below) -- the other gap D-039 flagged,
and the harder one. **Not cleanly closed at first**: doing this for real
surfaced a genuine robustness gap the LOO evaluation above never could,
because LOO only ever evaluates against more of the same kind of "arm at
rest" capture. A real episode's second goal renders *after* G1 has
already reached for the first one (and, on success, teleported it into
the tray), so the frame looks different from every training/calibration
capture -- and the probe confidently (81%) misjudged a genuinely
destroyed object as present. CLIP's zero-shot judgment on the identical
frame got it right.

**Closed for real as of D-055**: `collect_arm_occluded_examples()` below
collects training examples in that same post-first-attempt state (not
just arm-at-rest), via `capture_episode_subprocess.py`'s
`--attempt-object` option, which replays a genuine `attempt_goal()` --
reach *and* teleport-on-success -- before capturing. Notably, a reach-only
version of this (arm moved, nothing teleported) did NOT reproduce D-054's
gap -- a probe trained on arm-at-rest data alone judged those examples
12/12 correctly, meaning the real culprit wasn't just the arm entering
frame, it was everything the first attempt actually changes about the
scene. Once the capture matched that fully, a probe trained on
arm-at-rest data alone reproduced D-054's exact 81% misjudgment on the
new examples, confirming the reproduction was faithful before trusting
any "fix" built on top of it. Adding those examples to the training set
fixes the live-loop misjudgment, verified across 5 held-out seeds/
conditions, each checked in its own fresh process per the D-022 render-
budget discipline (checking multiple episodes in one shared process gave
a false regression the first time, purely from that budget being
exceeded -- not a real probe failure). See D-055 in ai-notes/decisions.md
and test_dinov2_probe.py's TestLiveDecisionLoopMatchesOracle. This module
remains not promotion-ready -- promotion-readiness for representation
robustness claims generally is a broader question than one fixed gap.
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


def collect_arm_occluded_examples(
    n_present: int, n_absent: int, seed_start: int = 0,
    scene_variant: str = "kitchen_cabinet",
) -> list[tuple[np.ndarray, bool]]:
    """D-055 follow-up to D-054's finding: collects master_chef_can examples
    after a real attempt_goal() for potted_meat_can has already run (reach
    + teleport-to-tray on success) -- the same out-of-distribution condition
    the live decision loop's *second* goal actually sees (D-054), which
    collect_labeled_examples() above never captures (every one of its
    examples is arm-at-rest, nothing else moved). Not just an arm-reach:
    an earlier version of this function only reran the reach motion and it
    was NOT enough to reproduce D-054's failure (see D-055) -- the first
    goal's teleport-on-success also puts potted_meat_can in the tray, which
    is apparently also part of what makes the second goal's frame look
    different from training. `n_present` comes from
    `intervention_kind="none"` (nothing destroyed, arm still moves);
    `n_absent` from `intervention_kind="chef_can_destroyed"` (guaranteed
    fired by the time a 25-step reach finishes, since onset_step_range is
    (2, 3)). Only makes sense for "kitchen_cabinet" -- _REACH_CONFIGS is
    only calibrated for that scene's G1 base position (see
    tidy_up_env_replicacad_humanoid.py)."""
    cfg = _OBJECT_VISUAL_CONFIG[scene_variant]["master_chef_can"]
    y0, y1, x0, x1 = cfg.crop
    examples: list[tuple[np.ndarray, bool]] = []
    plan = [("none", n_present)] + ([("chef_can_destroyed", n_absent)] if n_absent > 0 else [])
    seed = seed_start
    for intervention_kind, count in plan:
        for _ in range(count):
            out_path = Path(f"/tmp/_repr_capture_arm_{seed}.npz")
            subprocess.run(
                [
                    sys.executable, str(_CAPTURE_SCRIPT),
                    "--seed", str(seed), "--out", str(out_path),
                    "--scene-variant", scene_variant,
                    "--intervention-kind", intervention_kind,
                    "--attempt-object", "potted_meat_can",
                ],
                check=True, capture_output=True,
            )
            data = np.load(out_path)
            frame = data["frame"]
            label = bool(data["exists_master_chef_can"])
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


def fit_probe(examples: list[tuple[np.ndarray, bool]]):
    """Fits a logistic-regression linear probe on DINOv2 embeddings of
    `examples` and returns it for real use afterward (D-054) -- unlike
    fit_and_evaluate_probe(), which only reports leave-one-out accuracy
    and discards the fit. Query it with
    `probe.predict(dinov2_embed(new_crop).reshape(1, -1))`."""
    from sklearn.linear_model import LogisticRegression

    embeddings = np.stack([dinov2_embed(crop) for crop, _ in examples])
    labels = np.array([label for _, label in examples], dtype=int)
    return LogisticRegression(max_iter=1000).fit(embeddings, labels)


def run_end_to_end_episode_dinov2(
    env, q_table: dict, probe, scene_variant: str = "kitchen_cabinet",
) -> dict:
    """DINOv2 counterpart to atr.pipeline.run_end_to_end_episode() (D-029/
    D-050): wires DINOv2 into a real live decision loop -- the harder of
    D-039's two flagged gaps. **First version not cleanly closed (D-054):**
    a real live episode surfaced a genuine robustness gap -- by the time
    this checks the *second* goal, G1's arm has already moved (reaching
    for the first goal, and if that succeeded, the first object is now in
    the tray too), so the frame it renders looked out-of-distribution
    relative to every training/calibration capture at the time (all taken
    with the arm at rest) -- the probe misclassified a genuinely destroyed
    object as present, confidently (81%). CLIP's zero-shot judgment on the
    exact same frame got it right. This function's own logic was left
    exactly as a faithful, direct port of run_end_to_end_episode()'s
    structure throughout -- the fix (D-055) is entirely in what the probe
    passed in is trained on (see collect_arm_occluded_examples()), not in
    tuning this function's crop or decision logic to force one case to
    pass. See D-054/D-055 in ai-notes/decisions.md and
    TestLiveDecisionLoopMatchesOracle in test_dinov2_probe.py.

    Otherwise identical integration to the CLIP version: parsed
    instruction, a trained Q-table decides attempt vs. skip via the same
    greedy_action() lookup, real arm motion executes it.

    Scoped to `master_chef_can` only, unlike the CLIP version, which
    checks both goals -- not an oversight. `potted_meat_can` never goes
    absent under this env's `chef_can_destroyed` intervention (see
    collect_labeled_examples()'s own docstring), so there are no
    negative examples anywhere in this project to fit a present/absent
    probe against for it. Treating "no visual check possible for an
    object that's never actually intervened on in this scenario" as
    always-feasible is honest here -- it's what oracle_feasibility would
    report too, not a shortcut around a real gap. Fabricating negative
    examples or skipping this distinction silently would be the actual
    shortcut."""
    from atr.pipeline import _instruction_graph
    from atr.envs.tidy_up_replicacad_humanoid_policies import _TRAY_SLOTS, _summarize, attempt_goal
    from atr.policies.q_learning import SKIP, greedy_action

    cfg = _OBJECT_VISUAL_CONFIG[scene_variant]["master_chef_can"]
    y0, y1, x0, x1 = cfg.crop

    graph = _instruction_graph()
    per_goal = {}
    for i, goal in enumerate(graph.goals):
        if goal.target_object == "master_chef_can":
            frame = env.render()[0].cpu().numpy()
            crop = frame[y0:y1, x0:x1]
            embedding = dinov2_embed(crop).reshape(1, -1)
            perceived_feasible = bool(probe.predict(embedding)[0])
        else:
            perceived_feasible = True  # never intervened on here -- see docstring

        key = (goal.id, perceived_feasible)
        action = greedy_action(q_table, key)

        if action == SKIP:
            per_goal[goal.id] = {
                "achieved": False, "steps_used": 0, "skipped": True,
                "perceived_feasible": perceived_feasible,
            }
        else:
            result = attempt_goal(env, goal, _TRAY_SLOTS[i])
            result["perceived_feasible"] = perceived_feasible
            per_goal[goal.id] = result

    return _summarize(per_goal)
