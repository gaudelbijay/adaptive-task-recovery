# AI Project Notes

This directory contains short, frequently updated notes used to track project
execution. Stable design and research documentation lives in [`../docs`](../docs/).
The current project direction is feasibility-aware vision-language RL after
irreversible world changes; historical humanoid-recovery decisions are marked
as superseded rather than silently removed.

## Files

- [decisions.md](decisions.md) — important decisions and their rationale
- [issues_and_risks.md](issues_and_risks.md) — active blockers, uncertainties, and risks
- [review-request-task-schema.md](review-request-task-schema.md) — standing request for teammate review of D-013's task schema and everything built on it since; see that file's own "why now"
- `status.md`, `todo.md`, `recent_changes.md` — **superseded, 2026-08-01**,
  each now a pointer to the root [`../STATUS.md`](../STATUS.md), which
  covers current phase/focus, the todo table, and the change log in one
  place. Kept as stubs (not deleted) so old links still resolve. Do not
  revive these as a second tracker — that's exactly how they went stale
  the first time (unmaintained since 2026-07-26 while `STATUS.md` was the
  one actually kept current).

## Maintenance rules

- Update the root `STATUS.md` (phase, todo table, change log) whenever the
  active phase, focus, or todo state changes — not the `ai-notes/` stubs
  above.
- Record decisions that would be expensive or confusing to revisit in
  `decisions.md`.
- Close resolved risks instead of deleting their history.
- Use dates in `YYYY-MM-DD` format..
