"""Static, adaptive, hierarchical, and oracle policy baselines.

`baselines.py` (D-040) is env-agnostic policy-decision logic, promoted
from four near-identical `spikes/task_schema_draft/policy_baselines*.py`
copies (one per embodiment). Each spike env file keeps its own, genuinely
different `attempt_goal()` (the low-level motion) and calls into these
functions with it as a parameter -- see `baselines.py`'s own docstring
for why this was a real duplication risk, not a hypothetical one.
"""

from __future__ import annotations
