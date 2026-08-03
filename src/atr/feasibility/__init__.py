"""Change and per-goal feasibility models.

`oracle.py` is D-013's reviewed core schema (Accepted, D-037) -- pure,
simulator-independent, privileged-state feasibility and constraint-
violation checking (`goal_feasible`, `goal_achieved`,
`goal_dependencies_satisfied`, `constraint_violated`,
`evaluate_goal_graph`). A *learned* feasibility model (from pixels, not
privileged state) has not been promoted here yet -- the closest existing
evidence is `spikes/task_schema_draft/clip_feasibility.py` (D-020) and
`dinov2_probe.py` (D-023), still spike-stage.
"""

from __future__ import annotations
