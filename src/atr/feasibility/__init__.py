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
remains spike-stage in `spikes/task_schema_draft/` -- its own evidence
(one scene layout only, never wired into a live decision loop) hasn't
made its own promotion case yet.
"""

from __future__ import annotations
