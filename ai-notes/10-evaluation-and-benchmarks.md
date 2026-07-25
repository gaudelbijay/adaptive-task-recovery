---
title: Evaluation and Benchmarks
status: draft
last_updated: 2026-07-24
---

# Evaluation and Benchmarks

## 1. Headline metrics

| Metric | Definition | Computed against |
|---|---|---|
| Task success rate (nominal) | % episodes reaching task goal with no injected failure | Sanity check — should stay high across all system versions |
| Task success rate (under failure) | % episodes reaching task goal when a failure is injected mid-episode | Primary comparison metric across baselines |
| Recovery success rate | % of injected-failure episodes where the recovery layer restored task feasibility (task later succeeded) | Isolates recovery-layer contribution specifically |
| Time-to-recovery | Steps/seconds from failure onset to arbiter handing control back to the task policy | Reported as a distribution, not just mean — tails matter for real-world usability |
| Detection precision/recall/latency | Per [06-failure-taxonomy-and-detection.md](06-failure-taxonomy-and-detection.md) §4 | Failure monitor evaluated in isolation |
| Retry count | Number of recovery attempts per failure event before success/abort | Flags oscillation/thrashing, per [07](07-recovery-policy-design.md) §5 |
| Fall rate (real robot only) | % sessions/episodes ending in an uncontrolled fall | Hardware phase safety metric, tracked from session 1 |
| Sim2real gap | Metric delta (e.g., recovery success rate) between sim and hardware, same task/failure config | Hardware phase only |

Report every headline metric **broken down by failure type and severity**, not only aggregated — aggregate numbers are easy to game (e.g., doing great on easy failures, ignoring hard ones) and reviewers who've done robotics work will ask for the breakdown anyway.

## 2. Baselines (build all of these — comparisons are the actual contribution)

1. **No-recovery**: task policy runs unmodified; episode ends in whatever state the failure leaves it. This is your floor.
2. **Scripted/heuristic recovery**: a hand-coded behavior-tree-style fallback (e.g., "if grasp lost, retry grasp once from the same approach") — the realistic industry-standard comparison, not a strawman.
3. **Blind periodic retry**: no failure detector at all; the system just periodically attempts a generic "reset and retry" regardless of whether anything is actually wrong — isolates the value of *detection* specifically (compare against your full system to show detection matters, not just having a recovery skill).
4. **Monolithic robust policy**: a single policy trained under heavy domain randomization with no explicit detection/recovery modules — tests the core architectural bet from [03](03-system-architecture.md) §5 head-to-head.
5. **Oracle detector (upper bound)**: full recovery pipeline but using the simulator's ground-truth failure label instead of the learned detector — measures how much headroom is left in the detector specifically, separate from the recovery skills.

## 3. Ablations

- Detector: threshold-only vs. ensemble-dynamics vs. sequence-model (per [06](06-failure-taxonomy-and-detection.md)).
- Arbiter: rule-based vs. learned skill-selection (per [07](07-recovery-policy-design.md) §4), once v2 exists.
- Curriculum vs. no curriculum for recovery-skill RL training.
- Generalization: train on failure types {A, B}, evaluate on held-out type {C} — the specific test of RQ4 from [01](01-problem-statement-and-motivation.md).
- Severity generalization: train at low severity only, evaluate across the full severity range.

## 4. Statistical rigor

- Run every reported number across **multiple seeds** (minimum 3, more if compute allows) and report mean ± std or confidence intervals, not a single run's number — RL results are notoriously seed-sensitive, and reporting a single lucky run is one of the most common credibility gaps reviewers look for.
- Use a **fixed, held-out set of evaluation episodes/seeds** (not sampled fresh every eval) so comparisons across checkpoints/baselines are apples-to-apples.
- For hardware results, be explicit about small sample sizes (likely true given safety/time constraints) — report raw counts ("6/8 successful recoveries across 3 sessions"), don't dress up small-N hardware results as statistically rigorous.

## 5. Proposed open contribution (stretch)

Package the task suite + failure-injection API + baselines as a small, documented benchmark others could reuse (e.g., "Humanoid Task Recovery Benchmark on ManiSkill3") — even a modest, well-documented benchmark repo with a clear README and baseline numbers is a strong, concrete, linkable portfolio artifact distinct from the research results themselves, and is the kind of thing that's genuinely useful to cite in a resume/interview ("I built and open-sourced X").
