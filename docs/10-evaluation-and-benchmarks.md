---
title: Evaluation and Benchmarks
status: draft
last_updated: 2026-08-02
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

**Toy-scale first instances (2026-07-29, extended through 2026-08-01):**
"static language-conditioned policy" and "oracle-feasibility policies"
both have first hand-authored implementations in
`spikes/task_schema_draft/policy_baselines.py` (D-014) — a single
scenario, confirmed with the same result across four robot/scene
combinations since (D-016–D-018, D-021), not a benchmark run against the
other required baselines above. "Oracle-feasibility policy" also now has a
*learned* variant (`rl_policy.py`, D-025) — tabular Q-learning that
discovers the same policy from reward instead of having it hard-coded,
still using privileged/oracle feasibility as its input, so still not the
"full self-supervised feasibility-conditioned agent" this list's last
entry describes. None of the other required baselines (domain-randomized,
frame-difference detector, task-reward-only encoder, symbolic replanner)
have first instances yet.

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

**First real implementation, 2026-08-02 (D-042,
`src/atr/evaluation/harness.py`):** `compare_policies()` runs paired
seeds across policies and reports bootstrap confidence intervals per
metric — env/policy-agnostic, works against any TidyUp variant. Applied
immediately to H2's original static-vs-feasibility-aware comparison
(D-014): 30 paired seeds, canonical env, `bowl_destroyed` intervention.
Every interval collapsed to a single point — zero outcome variance
across all 30 seeds, for every metric, every policy. Reported as what it
is: this toy setup (fixed intervention, fixed onset window, fully
deterministic policies) has no real variance to measure at this scale,
confirmed not a harness bug via separate unit tests against known
distributions. The statistical machinery is real and ready; the current
comparisons just aren't stochastic enough yet to need it. Will start
mattering once either intervention timing is randomized across a window
that changes outcomes, or a perceptual policy (real CLIP/DINOv2 error
variance, not privileged-state ground truth) is what's compared.

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
