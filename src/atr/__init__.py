"""Committed project architecture.

D-013's core schema (`Goal`/`Constraint`/`GoalGraph`, oracle feasibility,
the intent guard) was accepted and promoted here 2026-08-02 (D-037) --
see `ai-notes/decisions.md` for the review resolution and
`ai-notes/review-request-task-schema.md` for what was actually asked.
Everything else that built on that schema (language parsing, vision-based
feasibility, self-supervised representations, the learned policy, the
end-to-end pipeline) remains evidence *for* the schema, not part of it,
and stays in `spikes/task_schema_draft/` until its own case for
promotion is made.
"""

from __future__ import annotations
