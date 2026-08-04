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

`tidy_up_env_replicacad.py` + `tidy_up_replicacad_policies.py` (D-048)
are the third variant: a real ManiSkill3 `ReplicaCADSetTableTrain`
apartment with a mobile Fetch robot, real YCB objects, and real
navigation (`navigation.py`, a generic grid + Dijkstra planner promoted
alongside since the policies file depends on it). Unlike D-046/D-047's
canonical/humanoid envs, this one has no `_OBJECT_SPECS`-style dict to
duplicate or derive from -- object positions come from the real scene
dataset, not hand-placed boxes -- so `_TRAY_POSITION`/`_TRAY_HALF_SIZES`
were already correctly imported (not copy-pasted) before promotion, and
`_LAST_KNOWN_POSITIONS` are legitimately standalone empirical fallbacks,
same role as `clip_feasibility.py`'s `_OBJECT_VISUAL_CONFIG`.

`tidy_up_env_replicacad_humanoid.py` + `tidy_up_replicacad_humanoid_policies.py`
(D-049) are the fourth and final variant: G1 fixed-base, placed (not
navigating) in the same real apartment as the Fetch variant. Same clean
pattern as D-048 -- real YCB objects, tray/fallback positions already
imported rather than duplicated, nothing to fix. This closes out all
four embodiment/scene variants named in docs/00's build-up order.

`capture_episode_subprocess.py` (D-052) is a standalone script (never
imported as a module, run via `subprocess.run([sys.executable,
str(_CAPTURE_SCRIPT), ...])`) that captures one render-producing reset of
the ReplicaCAD-Humanoid env in its own fresh process -- a real,
necessary workaround for D-022's confirmed upstream ManiSkill3 rendering
bug. Promoted despite its main caller
(`spikes/task_schema_draft/dinov2_probe.py`) not being promotion-ready,
same situation D-039 already handled for `device_utils.py`: this script
also serves the already-promoted `clip_feasibility.py`'s kitchen_sink
tests, and `dinov2_probe.py` depending on promoted code is the expected
direction, not a problem. Callers now locate it via its own module
`__file__` rather than a hardcoded relative path.
"""

from __future__ import annotations
