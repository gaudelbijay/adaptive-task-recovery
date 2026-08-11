"""Intent guard and violation monitors.

`intent_guard.py` is D-013's reviewed core schema (Accepted, D-037) --
`validate_action()`, the first toy test of H3
(docs/01-problem-statement-and-motivation.md). Blocks one specific
pattern (a candidate action targeting an object under a `never_move`
constraint that no real goal currently requires touching). R-010 in
`ai-notes/issues_and_risks.md` flagged that the original test only ever
exercised the easy, zero-recall-cost case; D-058 built the harder one --
confirmed the guard doesn't over-block a goal in direct conflict with a
matching constraint, and found (then fixed) an opposite-direction gap:
without privileged `state`, a conditional goal (`Goal.condition`)
exempted its target object even while its condition didn't hold.

D-082 reports recall and violation rate together. D-083 lets the guard check
predicted incidental effects, and `effect_predictor.py` (D-084) supplies the
first conservative producer using a straight-line swept corridor over object
centers. It is a screening model, not collision-accurate robot geometry.
D-085 extends that screening to every segment of a waypoint path.
D-086 adds optional object radii so the check is not limited to point centers.
`envs/navigation.py::screen_navigation_path()` (D-087) adapts real 2D planner
waypoints to this prediction-and-guard interface before execution.
"""

from __future__ import annotations
