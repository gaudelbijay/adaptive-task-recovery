---
title: Evaluation and Benchmarks
status: draft
last_updated: 2026-08-24
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
result: chance-or-worse LOO accuracy. The original run showed a fold-wise
majority-class pattern, but a fresh 2026-08-28 Jarvis capture fit the 12
training images strongly while still failing to generalize under LOO. The
reproducible claim is therefore poor held-out generalization, not universal
constant-output collapse. Both pretrained representations remain stronger on
this narrow comparison. See
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

The forced-classification-versus-abstention ablation now has an executable,
leakage-resistant evaluator (D-074): calibration counts and held-out correct
actions are passed separately to `compare_forced_vs_selective()`, which reports
forced risk, selective risk, and selective coverage. Its controlled regression
fixture verifies the metric and decision behavior but is not counted as a
benchmark result; the real simulator-backed wide-timing comparison is still
pending a renderer-capable runtime.

The real wide-timing version is now predeclared as a simulator test (D-075):
20 calibration episodes and 80 held-out episodes with disjoint seed ranges. It
deliberately expects the plausible negative H5 outcome—zero risk for both
methods but lower selective coverage—rather than changing the setup until
abstention wins. Execution is pending the full-suite CI's lavapipe runtime.

## Statistical protocol

Predeclare primary metrics and splits. Use paired episode seeds across methods,
bootstrap confidence intervals, and effect sizes. Correct or clearly label
multiple exploratory comparisons. Publish per-seed results and failure cases.

### Integrated learned-recovery protocol

The frozen `learned_recovery_ppo_v6` evaluation loads the training-selected
`best.pt` checkpoint and runs 256 deterministic held-out episodes per seed in
each of two conditions: intervention probability 1.0 and nominal probability
0.0. Evaluation seeds are disjoint from training and checkpoint selection.
Methods with the same training seed receive common random numbers, enabling a
paired bootstrap comparison over 768 episodes. Raw episode records are retained.

The primary endpoint is **safe success**: ordered task success and no
protected-object violation anywhere in the episode. Raw success and violation
rate remain separate secondary metrics. Results are also stratified by whether
the first requested goal or the second requested goal was physically removed;
this prevents a policy that only handles the easy ordering from appearing to
solve recovery. Nominal evaluation checks the cost of intervention training.
The result is not promoted until all nine checkpoints, 4,608 held-out episodes,
branch labels, and paired comparisons are complete.

**Scaled execution (D-125):** the frozen v1 matrix and cluster workflow
are specified in [`docs/11-scaled-experiments.md`](11-scaled-experiments.md).
Unlike the original in-process harness, it uses content-addressed cases,
stable paired shards, atomic per-episode artifacts, resume/retry, exact code and
runtime provenance, manifest-aware completeness checks, and stratified paired
aggregation. The full matrix is configured but not yet run; smoke evidence is
not reported as the primary result.

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

**Root cause found and fixed, 2026-08-07 (D-070):** the zero-variance
result above (and D-069's, after it) traced to `onset_step_range=(2, 3)`
-- `tidy_up_env.py` samples via `rng.integers(*self.onset_step_range)`,
and numpy's `Generator.integers()` is exclusive on the upper bound
(unlike Python's inclusive `random.randint`), so `(2, 3)` always samples
exactly `2`. Every prior comparison in this project used a range this
narrow or narrower. Fixed by using a genuinely wide range, `(10, 60)` --
wide enough to span both goals' own ~25-step attempt durations, not just
to vary the onset value itself; narrower ranges like `(5, 15)`/`(5, 40)`
still don't produce outcome variance. `static`/`feasibility_aware` both
got real, non-degenerate bootstrap CIs for the first time
(`tests/drafts/test_wide_onset_timing_variance.py`). Investigating why
`learned`'s result didn't fit that pattern surfaced a substantive,
unplanned finding of its own -- see D-070 in `ai-notes/decisions.md` and
docs/01's H2 update: under this wider, more realistic timing regime,
"currently feasible" stops reliably predicting "will complete" -- this
part held up under further scrutiny. D-070's further claim that the
trained Q-value itself was "the mathematically correct" response did not
(**corrected 2026-08-08, D-071**): verifying a calibrated version of the
same quantity against the Q-table's own point estimate surfaced that the
Q-value was trained on a *pooled* state (across every `intervention_kind`,
not just the risky one), and a proper bootstrap CI on that pooled quantity
straddles zero (`n=441`) -- genuinely ambiguous, not confidently negative.
Only the quantity conditional on the risky intervention actually being
active is robustly negative (CI excludes zero, `n=198`). This is exactly
the failure mode this section's own protocol exists to catch (a
single-run point estimate stood in for a confidence interval, D-070's
original miss) -- caught here by applying the same `bootstrap_ci()`
machinery to a *training* signal, not just a final policy comparison, a
first for this project. See D-071 for the fix (calibrate per
`(goal_id, intervention_kind)` instead of pooling,
`src/atr/feasibility/calibrated_feasibility.py`) and docs/01's H2 entry
for the corrected research implication.

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
