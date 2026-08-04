"""Tasks, interventions, and oracle-feasibility-label wiring to real
ManiSkill3 scenes.

`tidy_up_env.py` (D-045) is the canonical task env: five objects on a
tabletop, a matched irreversible/reversible intervention pair
(`bowl_destroyed`/`temporary_obstacle`, per docs/04's explicit "include
matched reversible and temporary changes" requirement), wiring the
promoted schema (`atr.language.goal_graph`, `atr.feasibility.oracle`) to
real privileged state. `tidy_up_policies.py` (D-046) is that env's own
policy-facing API: `attempt_goal()` (real arm motion, genuinely
env-specific) plus thin `static_policy`/`feasibility_aware_policy`/
`naive_substitution_policy` wrappers over `atr.policies.baselines`
(D-040). The three other embodiment/scene variants
(`spikes/task_schema_draft/tidy_up_env_humanoid.py`/`_replicacad.py`/
`_replicacad_humanoid.py`, each with their own `policy_baselines_*.py`)
remain spike-stage -- each needs its own promotion case, same discipline
every promotion since D-037 has followed.
"""

from __future__ import annotations
