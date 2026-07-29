---
title: Evaluation and Benchmarks
status: draft
last_updated: 2026-07-26
---

# Evaluation and Benchmarks

## Primary outcomes

| Outcome | Definition |
|---|---|
| Feasible-goal completion | Fraction/value of oracle-feasible goals achieved |
| Intent violation rate | Episodes with any hard constraint or invalid substitution — first toy measurement: `dont_move_glass_violated` in `spikes/task_schema_draft/policy_baselines.py`'s `naive_substitution_policy` (D-015) |
| Adaptation regret | Valid goal value gap from oracle-feasibility policy |
| Feasibility quality | Per-goal discrimination, calibration, and selective risk |
| Adaptation latency | Steps after change before strategy switches appropriately |
| Efficiency | Steps/resources spent on goals already oracle-infeasible — first toy measurement: `wasted_steps` in `spikes/task_schema_draft/policy_baselines.py` (D-014) |
| Humanoid execution success | Chosen semantic skills completed without safety failure |

Report nominal task performance separately so adaptation gains cannot hide a
weaker base policy. Report achieved goals, infeasible goals, abandoned-feasible
goals, and violations rather than collapsing them into one reward.

Decompose end-to-end failure into perception, feasibility, high-level strategy,
intent guard, and humanoid skill execution. Also report a high-level oracle-skill
evaluation so controller reliability does not obscure the research result.

## Required baselines

- no-change upper control;
- static language-conditioned policy;
- domain-randomized policy without explicit feasibility;
- simple frame-difference change detector plus rules;
- adaptive policy with task-reward-only visual encoder;
- pretrained frozen and fine-tuned visual encoders;
- symbolic replanner with learned state;
- oracle-state and oracle-feasibility policies;
- full self-supervised feasibility-conditioned agent with intent guard.

**Toy-scale first instances (2026-07-29):** "static language-conditioned
policy" and "oracle-feasibility policies" both have first hand-authored
implementations in `spikes/task_schema_draft/policy_baselines.py` (D-014) —
a single scenario, not a benchmark run against the other required baselines
above.

## Core ablations

- remove temporal history;
- remove feasibility head;
- remove intent guard;
- replace factorized goal graph with one instruction embedding;
- image-only versus temporal/object-centric self-supervision;
- frozen versus fine-tuned encoder;
- forced classification versus calibrated abstention;
- seen versus held-out goal-change combinations.

## Statistical protocol

Predeclare primary metrics and splits. Use paired episode seeds across methods,
bootstrap confidence intervals, and effect sizes. Correct or clearly label
multiple exploratory comparisons. Publish per-seed results and failure cases.

## Critical controls

- unchanged worlds, to measure unnecessary adaptation;
- temporary/reversible changes, to measure premature abandonment;
- visually salient but feasibility-neutral changes;
- visually subtle but feasibility-changing interventions;
- paraphrases with identical formal goals;
- adversarial reward cases where an intent violation is tempting.
- matched evaluation with oracle skill outcomes versus actual humanoid execution.

## Claim boundary

Results support claims only within the benchmark's operational definitions of
feasibility and intent. Generalization to real robots, arbitrary instructions,
or human values requires separate evidence.
