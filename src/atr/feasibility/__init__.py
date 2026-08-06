"""Change and per-goal feasibility models.

`oracle.py` is D-013's reviewed core schema (Accepted, D-037) -- pure,
simulator-independent, privileged-state feasibility and constraint-
violation checking (`goal_feasible`, `goal_achieved`,
`goal_dependencies_satisfied`, `constraint_violated`,
`evaluate_goal_graph`).

`clip_feasibility.py` (D-020/D-027, promoted D-039) is the first real
*perceptual* feasibility model: zero-shot CLIP judges object presence
from a rendered frame instead of privileged state, wired into the real
end-to-end decision loop (D-029). **Read its module docstring before
trusting it as general** -- its evidence is per-object/per-scene
calibration, not generalization, a meaningfully different (weaker) claim
than `oracle.py` or `instruction_parser.py`'s evidence.

`dinov2_probe.py` (D-023, self-supervised, no calibrated prompt needed)
remains spike-stage in `spikes/task_schema_draft/` -- tested on 2 scene
layouts now (D-053) and wired into a real live decision loop (D-054/
D-055, including a found-and-fixed robustness gap), but a closed gap in
one scenario isn't a general promotion-readiness claim on its own.

`frame_diff.py` (D-063) is the "simple pixel-difference change detector
plus rules" required baseline (docs/10) -- zero learned parameters, zero
supervision, just a thresholded pixel-difference score on the same
calibrated crops `clip_feasibility.py` uses. Weaker separation than
either CLIP or DINOv2 on the one scenario measured so far (real: destroyed
scores ~1.8x the survivor's, not either model's near-100% margins) --
exists to test whether their added complexity earns its keep, not to
replace them.

`task_reward_encoder.py` (D-066) is the "pixels trained only through
task reward" baseline H1's own wording (docs/01) actually asks for --
neither CLIP nor DINOv2 qualifies, since both start from a large
pretrained backbone (language-supervised, self-supervised). A small
conv encoder, no pretraining at all, trained from scratch on the same
~24-example toy dataset CLIP/DINOv2 were evaluated against. Measured
result, root-caused not just reported: it fails to learn any real
visual discrimination -- its output is a near-constant regardless of
image content (confirmed directly), converging to the training fold's
own majority class instead of anything about the image. The most direct
piece of evidence for H1's actual comparative claim in this project so
far: the pretrained representations both succeed at 100% LOO accuracy
on this data; training from scratch on the identical data does not
learn to discriminate at all.
"""

from __future__ import annotations
