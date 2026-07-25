---
title: Experiment Log Template
status: draft
last_updated: 2026-07-24
---

# Experiment Log Template

Keep a running log of every experiment that produces a number you might report anywhere (writeup, resume, interview). This is what makes claims defensible later — "I remember it worked well" is not a substitute for a dated row with a commit hash.

## How to use this file

- Append a new row per meaningful run (not every hyperparameter tweak — use judgment, but err toward logging more rather than less for anything tied to a phase's exit criteria in [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md)).
- Never edit a past row's result after the fact — if a result later turns out wrong (bug found, etc.), add a new row noting the correction and why, so the log stays an honest history rather than a cleaned-up story.
- Copy the table below into a real log (e.g., `EXPERIMENTS.md` at repo root, or a spreadsheet/W&B project) once implementation starts — this file is the template/instructions, not the log itself.

## Log table

| Date | Exp ID | Phase | Hypothesis | Config / commit hash | Steps/sec, GPU mem | Result summary | Next step |
|---|---|---|---|---|---|---|---|
| 2026-08-01 | 0001 | Phase 0 | Imported humanoid can hold a standing pose under PD control | `abc1234`, `configs/smoke/stand.yaml` | 12k steps/s @ 512 envs, 6GB | Stable for 30s across 100 seeds w/ ±5° init noise | Move to Phase 1 baseline task |

## Fields explained

- **Phase**: which phase from [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md) this belongs to.
- **Hypothesis**: what you expected before running it — write this *before* looking at results, even informally; it's the difference between an experiment and a fishing expedition, and it's what makes a "why did you try that" interview question answerable.
- **Config / commit hash**: exact reproducibility pointer — config file path plus git commit, not a vague description.
- **Result summary**: the actual numbers (precision/recall, success rate, steps/sec — whatever's relevant), not just "worked" or "didn't work."
- **Next step**: what this result changed about the plan — this is what turns a log into a narrative you can tell later ("run 0007 showed X, which is why I switched to Y").

## Session/incident notes (for anything safety- or stability-adjacent)

Even though this project is simulation-only, log any simulator instability incidents (physics blowups, NaN states, environments that silently produce wrong contact behavior) the same way a hardware near-miss would be logged — these are exactly the "I hit X, diagnosed it as Y, fixed it with Z" stories that make a project writeup credible, and they're easy to forget the details of if not written down close to when they happened.
