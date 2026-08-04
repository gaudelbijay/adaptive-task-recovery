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
(D-040).

`tidy_up_env_humanoid.py` + `tidy_up_humanoid_policies.py` (D-047) are
the same pairing for the Unitree G1 humanoid variant -- same schema,
different embodiment (joint-space reach instead of Cartesian IK, since
this robot has no Cartesian end-effector controller). Checked, not
assumed, before promoting: its `_TRAY_POSITION` z doesn't match
`_OBJECT_SPECS`'s spawn z the way the canonical env's did (D-046) --
this env's own `evaluate()` explains objects settle to a different real
height than their spawn height, so the two numbers are legitimately
different, not a stale duplicate. Left as-is rather than "fixed" to
match.

The two remaining embodiment/scene variants
(`spikes/task_schema_draft/tidy_up_env_replicacad.py`/
`_replicacad_humanoid.py`, each with their own `policy_baselines_*.py`)
remain spike-stage -- each needs its own promotion case, same discipline
every promotion since D-037 has followed.
"""

from __future__ import annotations
