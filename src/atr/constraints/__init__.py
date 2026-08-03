"""Intent guard and violation monitors.

`intent_guard.py` is D-013's reviewed core schema (Accepted, D-037) --
`validate_action()`, the first toy test of H3
(docs/01-problem-statement-and-motivation.md). Still narrow: blocks one
specific pattern (a candidate action targeting an object under a
`never_move` constraint that no real goal requires touching) -- see
R-010 in `ai-notes/issues_and_risks.md` for the harder recall/safety
trade-off this doesn't yet test.
"""

from __future__ import annotations
