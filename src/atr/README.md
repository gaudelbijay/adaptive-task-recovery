# `src/atr/`

Committed project architecture. **Currently empty on purpose** -- not
unfinished, not forgotten.

## Why empty

Everything the project has actually built so far lives in
[`spikes/task_schema_draft/`](../../spikes/task_schema_draft/), by design
(see D-006 and D-013 in [`ai-notes/decisions.md`](../../ai-notes/decisions.md)):

- D-006 says no simulator-specific architecture gets committed here until
  the simulator selection question (I-003) is actually settled.
- D-013's goal-graph schema -- the foundation everything else (language
  parsing, vision-based feasibility, the learned policy, the end-to-end
  pipeline) is built on -- is explicitly **proposed, not accepted**. It's
  currently out for review; see
  [`ai-notes/review-request-task-schema.md`](../../ai-notes/review-request-task-schema.md).

Moving code here before that review lands would answer the review's own
central question ("is this ready to become committed architecture?") by
fiat, which defeats the point of asking it.

## What this directory is for

Once the schema review resolves (accepted as-is, accepted with changes,
or sent back for rework), the reviewed pieces of `spikes/task_schema_draft/`
move here as the real, versioned, `src/`-layout package -- alongside the
sibling directories this same scaffolding pass added:
[`configs/`](../../configs/) for experiment configuration and
[`data/`](../../data/) for datasets, matching the existing
[`scripts/`](../../scripts/) and [`tests/`](../../tests/).

## What this directory is not

Not a second copy of the spike code, not a place to duplicate anything
while the review is pending. If you're looking for the actual working
implementation today, it's in `spikes/task_schema_draft/`.
