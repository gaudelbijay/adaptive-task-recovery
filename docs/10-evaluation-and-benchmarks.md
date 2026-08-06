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
still using privileged/oracle feasibility as its input. **"Simple
frame-difference change detector plus rules" built 2026-08-06 (D-063,
`src/atr/feasibility/frame_diff.py`):** zero learned parameters, same
calibrated crops CLIP uses. Real measured result on one scenario
(kitchen_cabinet, `chef_can_destroyed`, seed=0, confirmed reproducible
across 5 reruns): destroyed object scores ~1.8x the survivor's — a real
but much weaker separation than CLIP's or DINOv2's near-100% margins,
testing whether their added complexity earns its keep rather than
replacing them. **"Full self-supervised feasibility-conditioned agent
with intent guard" built 2026-08-06 (D-064,
`run_end_to_end_episode_dinov2_with_intent_guard()` in
`spikes/task_schema_draft/dinov2_probe.py`):** closes this list's last
entry — not new capability, DINOv2 perceptual feasibility (D-054/D-055)
plus a substitution attempt on the graph's own never-move-constrained
object plus the intent guard blocking it, combined for the first time.
Verified both directions: guarded run avoids the constraint violation
(zero wasted steps), unguarded run on the identical episode actually
violates it — confirming the guard's block is doing real work, not
passing vacuously. **"Domain-randomized policy without explicit
feasibility" built 2026-08-06 (D-065,
`src/atr/policies/domain_randomized.py`):** same domain-randomized
training loop `q_learning.py` already uses, minus the feasibility bit in
the state key — the policy has no way to perceive whether the current
episode's goal is actually feasible. Predicted the result from this
project's own reward shape before training, then confirmed it on the
real trained table: a goal that's only feasible half the time has
negative expected value to attempt blindly, so the policy learns to skip
it unconditionally. Measured consequence on two live episodes: matches
`feasibility_aware_policy` exactly when the goal really is infeasible
(costs nothing), but wrongly skips a genuinely achievable goal when it
isn't (`goals_achieved` drops from 2 to 1) — a real, measured recall
cost `feasibility_aware_policy` doesn't pay, since it can actually tell
the two cases apart. **"Adaptive policy with task-reward-only visual
encoder" built 2026-08-06 (D-066,
`src/atr/feasibility/task_reward_encoder.py`):** the baseline H1's own
wording actually asks for ("pixels trained only through task reward") —
a small conv encoder, no pretraining at all, trained from scratch on the
identical toy-scale data CLIP/DINOv2 were evaluated against. Measured
result, root-caused not assumed: 0% LOO accuracy, from a confirmed
majority-class collapse (every held-out example in every fold gets the
identical output regardless of image content) rather than noisy
guessing — real gradient flow and real weight changes were checked
directly, ruling out a training bug. The clearest evidence for H1's
comparative claim in the project so far: both pretrained representations
reach 100% LOO accuracy on this data; training from scratch on the same
data doesn't learn to discriminate at all. See
docs/01-problem-statement-and-motivation.md's H1 entry for the full
writeup. **"Symbolic replanner with learned state" built 2026-08-06
(D-067, `src/atr/policies/symbolic_replanner.py`):** unlike every other
policy here (one fixed pass through `graph.goals` in order), `plan()`
genuinely searches over goal orderings using `Goal.priority`/
`Goal.depends_on` -- schema fields defined since D-013 but never
actually used to choose a plan before this, only to gate a fixed order.
Verified on `dependent_goals_example()`'s real ordering constraint
(a higher-priority goal that depends on a lower-priority one being
achieved first) and end-to-end on a live episode, both with privileged
state and with real CLIP perception as the feasibility estimate --
"learned state" is not a claim the function can't be given oracle state
too, just that it doesn't require it. **"Pretrained frozen and fine-
tuned visual encoders" built 2026-08-06 (D-068,
`fit_finetuned()`/`fit_and_evaluate_finetuned()` in
`spikes/task_schema_draft/dinov2_probe.py`), closing this list
entirely:** unfreezes DINOv2's last transformer block and trains it plus
a linear head end-to-end, compared against the existing frozen-backbone-
plus-probe approach on the identical LOO set — both reach 100% (no
headroom either way at this scale, no overfitting cost observed either,
confirmed via a direct weight-change check). The more informative
measurement: neither the frozen probe nor the fine-tuned encoder gets
D-054's out-of-distribution arm-occluded case right when both are
trained on the same narrow (arm-at-rest-only) data — fine-tuning the
backbone doesn't substitute for D-055's actual fix (training data that
covers the real deployment distribution). See D-068 in
`ai-notes/decisions.md`.

**All required baselines above now have a first instance.**

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
