"""Instruction schema, parsing, and goal graphs.

`goal_graph.py` is D-013's reviewed core schema (Accepted, D-037) --
`Goal`, `Constraint`, `GoalGraph`. The controlled-grammar parser that
compiles instruction text into a `GoalGraph`
(`spikes/task_schema_draft/instruction_parser.py`, D-019/D-026) has not
been promoted here yet -- it's evidence for the schema, not part of it,
per `ai-notes/review-request-task-schema.md`.
"""

from __future__ import annotations
