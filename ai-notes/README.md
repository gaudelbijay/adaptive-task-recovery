# AI Project Notes

This directory contains short, frequently updated notes used to track project
execution. Stable design and research documentation lives in
[`../docs`](../docs/), indexed by [`../docs/README.md`](../docs/README.md).

The work centres on auditing whether a recovery benchmark measures recovery or
a shortcut, and on the recovery architecture evaluated against that audit.
Earlier directions — humanoid recovery, and the vision-language line that
preceded the audit — are marked superseded rather than silently removed, so the
record shows what was tried and why it was set aside.

## Files

- [decisions.md](decisions.md) — important decisions and their rationale
- [issues_and_risks.md](issues_and_risks.md) — active blockers, uncertainties, and risks
- [review-request-task-schema.md](review-request-task-schema.md) — standing request for teammate review of D-013's task schema and everything built on it since; see that file's own "why now"
- `status.md` — the living tracker: phase, todo table, and change log.
  Moved here from the repository root on 2026-09-02.
- `todo.md`, `recent_changes.md` — **superseded, 2026-08-01**, each a
  pointer to [`status.md`](status.md), which
  covers current phase/focus, the todo table, and the change log in one
  place. Kept as stubs (not deleted) so old links still resolve. Do not
  revive these as a second tracker — that's exactly how they went stale
  the first time (unmaintained since 2026-07-26 while `status.md` was the
  one actually kept current).

## Maintenance rules

- Update `status.md` (phase, todo table, change log) whenever the
  active phase, focus, or todo state changes — not the `ai-notes/` stubs
  above.
- Record decisions that would be expensive or confusing to revisit in
  `decisions.md`.
- Close resolved risks instead of deleting their history.
- Use dates in `YYYY-MM-DD` format.
