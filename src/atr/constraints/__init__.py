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
"""

from __future__ import annotations
