"""Instruction schema, parsing, and goal graphs.

`goal_graph.py` is D-013's reviewed core schema (Accepted, D-037) --
`Goal`, `Constraint`, `GoalGraph`. `instruction_parser.py` (D-019/D-026,
promoted D-038) is the controlled-grammar parser that compiles
instruction text into a `GoalGraph` -- reproduces every hand-authored
`GoalGraph` in this project from its own instruction text, and
generalizes to held-out paraphrases and a held-out object composition
(see `ai-notes/decisions.md` D-038 for the promotion case).
"""

from __future__ import annotations
