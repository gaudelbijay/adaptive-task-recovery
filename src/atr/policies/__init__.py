"""Static, adaptive, hierarchical, and oracle policy baselines.

`baselines.py` (D-040) is env-agnostic policy-decision logic, promoted
from four near-identical `spikes/task_schema_draft/policy_baselines*.py`
copies (one per embodiment). Each spike env file keeps its own, genuinely
different `attempt_goal()` (the low-level motion) and calls into these
functions with it as a parameter -- see `baselines.py`'s own docstring
for why this was a real duplication risk, not a hypothetical one.

`q_learning.py` (D-025/D-041) and `imitation.py` (D-060) learn the same
attempt/skip decision two different ways -- reward-driven Q-learning and
demonstration-driven behavioral cloning, over the identical
`(goal_id, feasible) -> {SKIP, ATTEMPT}` state/action space -- so they can
be trained and compared under matched conditions. See docs/07-adaptive-
policy-design.md and imitation.py's own module docstring for what the
comparison actually shows.
"""

from __future__ import annotations
