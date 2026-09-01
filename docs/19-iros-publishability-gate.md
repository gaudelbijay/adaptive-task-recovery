---
title: IROS Publishability Gate and Method Reset
status: active-experiment
last_updated: 2026-08-30
---

# IROS publishability gate

> **Naming.** V36--V60 are *visual* controller candidates; the A+ V1--V10 are
> *router* candidates. A counter is a later candidate, not a better one. See
> [`31-naming-and-identifier-key.md`](31-naming-and-identifier-key.md).


The dual-specialist RGB controller is a strong custom-benchmark result, not yet a strong
main-track IROS paper. Repository scale and the number of attempted variants do
not substitute for novelty or external validity. The visual robustness-patching line is closed: its final candidate retained
only 69.53% nominal safe success on seed 9351.

## Blocking weaknesses

1. The benchmark contains one intervention mechanism, two colored cubes, one
   camera, and one robot arm.
2. The incumbent controller is a privileged dual-teacher distillation recipe
   rather than a concise new recovery algorithm.
3. The winning actor fails small camera and appearance shifts.
4. There is no held-out failure-mechanism result and no shared external
   benchmark comparison.
5. Hundreds of millions of interactions and many post-hoc variants make a
   sample-efficiency or clean-ablation claim difficult.

## New method direction: irreversible feasibility memory

The next method will separate perception of goal loss from motor control.

- A frozen self-supervised visual backbone compares an episode reference frame
  with the current frame using goal-conditioned features.
- A learned head predicts a three-state belief for each requested goal:
  pending, completed, or physically unavailable.
- An irreversible temporal filter may latch `unavailable` only after calibrated
  evidence and may not silently revert it. Uncertainty causes continued
  observation or abstention, not an unqualified expert switch.
- A modular controller consumes the belief and composes nominal and recovery
  experts. The visual belief, not an evaluator label, performs deployment-time
  routing.
- Training uses factual/counterfactual paired renders and privileged labels,
  disclosed as supervision. The primary novelty claim is causal feasibility
  estimation and policy composition, not pure pixel RL.

The first fail-fast experiment is
`scripts/probe_v3_goal_loss_dinov2.py`. It trains a canonical-view linear head
on reference/current DINOv2 features and evaluates unchanged weights under four
renderer-native camera/lighting profiles. No controller allocation is allowed
unless this perception-only probe materially exceeds the matched low-resolution
pixel baseline on every profile.

## Required paper-level evidence

A method is paper-eligible only if all of the following are frozen before the
corresponding outcomes:

1. At least two physically distinct goal-loss mechanisms, with one held out
   from method training.
2. At least three goal identities or a variable-length goal sequence; the
   current red/blue ordering alone is insufficient.
3. Nominal, strict removal, each removal position, protected-object violation,
   and recovery latency reported for the same checkpoints.
4. Three or more independent training seeds and hierarchical uncertainty.
5. Oracle-belief, end-to-end dual-specialist RGB, no-memory, random-feature, and
   no-counterfactual-pair baselines under identical evaluation seeds.
6. A reserved camera/lighting/object-appearance suite with no tuning on its
   outcomes.
7. A second ManiSkill task family or a small real-robot validation. Without
   either, the claim must be benchmark-focused rather than a general method
   claim.
8. Interaction, supervision, backbone, and teacher costs reported explicitly.

The IROS manuscript remains an evidence draft until this gate passes. Good
writing cannot repair a failed gate, and no threshold may be relaxed after an
outcome is observed.

## 2026-08-31 reconciliation after the temporal-composition study

The assessment above predates the completed A+ V3 experiment. The full-geometry centered router fixed the input mismatch and produced a
positive composition result: that model reached 100% held-out reverse accuracy offline while the static and
unstructured controls reached 3.22% and 0%; on the once-only closed-loop
confirmation it achieved 573/576 held-out reverse safe successes and 2655/2880
overall, versus 2369/2880 for the matched unstructured baseline. The overall
gain was 9.93 points with a 95% Newcombe interval of [7.54, 12.29] points.

It remains rejected because its nominal condition reached only 456/576
(79.17%), below the frozen 82% worst-condition floor. The six router candidates
that followed retained that rejection history and used fresh selection/confirmation families; none was
reinterpreted as a pass.

The guarded factorized dispatch router closes the custom-benchmark method
blocker. On the once-only untouched `347000000` confirmation family, it achieved 2655/2880 (92.19%)
safe success with 0.83% violations, versus 2354/2880 (81.74%) for the strongest
matched non-oracle baseline. The +10.45-point gain has a 95% Newcombe interval
of [8.05, 12.83] points. It also passed the frozen pooled OOD gate at 6369/7680
(82.93%). This result has a real boundary: twelve-step control delay reached
only 55.83%, and the 48-step temporary-block axis produced 15.83% violations.

Passing the guarded factorized dispatch router's gate (router series V10) is
necessary but not sufficient for a top-tier claim. The remaining
blocker is closed-loop external validity. The second-family protocol
is frozen separately in
`configs/a_plus_external_peg_insertion_gate_v1.json`: an intervention extension
of ManiSkill3's official `PegInsertionSide-v1`, chosen because contact-rich
six-DoF insertion is physically distinct from planar cube transport. Runtime
pose assignment and the older teleport-based TidyUp controllers are forbidden.
The external gate uses matched observations, shared specialists, a held-out
ejection direction, three seeds, a static shortcut ceiling, and a once-only
confirmation bank.

The release decision is conjunctive:

1. Its primary and pooled OOD gates pass on `347000000` (completed).
2. The external PegInsertion gate passes on `429000000`.
3. The existing REBOOT result remains labeled offline real-robot transfer, not
   closed-loop real-robot control.
4. Costs and all rejected V-series candidates are compressed into auditable
   tables rather than presented as independent discoveries.

Until both closed-loop gates pass, the defensible paper framing is benchmark
and representation evidence, not a top-tier general recovery method.

## Correction: what the router comparison actually establishes

Three corrections apply to the paragraphs above. None changes a frozen
threshold or reinterprets a rejection; all are recorded because the original
scoring was incomplete or the wording overclaimed.

**The declared comparison was incomplete.** The gate listed five methods.
`heuristic_v28_router` and `oracle_mechanism_router_upper_bound` had no
implementation when the router was scored, so "strongest matched non-oracle"
ranged over three arms, not five. Both are now built and run on the same
family.

**The reported gain included an unmatched mechanism.** The causal arm ran a
factorized sweep dispatch that no other arm can execute. Disabling it leaves
held-out reverse ejection at exactly 561/576 and changes only forward ejection.
The matched gain over the unstructured GRU is 7.26 points [5.44, 9.07] -- still
above the frozen five-point floor, but the correct number to quote is 7.26, not
10.45.

**"Causal composition" is the wrong description of the held-out result.** The
hand-written motion-threshold heuristic reaches the *same* 97.40% on held-out
reverse ejection as the learned model. A held-out mechanism that a six-branch
threshold rule identifies perfectly is not evidence of learned causal
composition. The learned model's advantage over that heuristic is confined to
temporary-versus-permanent obstruction (84.38% against 0.00%) and to
constraint violations (0.83% against 16.98%).

The history ablation supports the same narrowing. Removing geometry history
collapses held-out accuracy to 0.000 on every seed, but reversing the prefix in
time leaves it at 97.7%, 77.6%, and 96.9%. History is necessary; its direction
largely is not. Throughout this repository "causal" should be read as
*temporally causal* -- no access to future frames -- and not as causal-dynamics
inference. Class and checkpoint identifiers keep the older name because frozen
gate hashes and confirmation provenance key on those exact strings.

The revised blocking weakness is therefore sharper than the original list: the
benchmark's held-out mechanism is too easy to detect, so mechanism recognition
cannot carry a method claim. Persistence disambiguation can, and that is the
result the external PegInsertion gate should be designed to test.

## Revised paper claim after the completed arm set

The single-observation control changes what this work can claim, and sharpens
it. A memoryless MLP reading one past frame reaches 100% held-out reverse
accuracy offline and 97.40% closed-loop -- identical to the causal router and
to the hand-written threshold rule. Held-out mechanism composition is therefore
not a finding; three methods including a one-frame model solve it perfectly.

What survives is a mechanism result with a clean ablation signature. The
permanent/temporary confusion pair is the only place memory matters, and the
two non-recurrent arms fail it in opposite directions: the threshold rule
commits to permanent (100.00% / 0.00%), the one-frame model commits to
temporary (0.00% / 84.38%). Both recurrent arms solve both sides. That is a
falsifiable claim with a mechanism, not a leaderboard margin.

The paper this supports is a benchmark-and-mechanism study, not a method paper:
recovery benchmarks admit shortcuts at two levels -- instantaneous geometry
identifies the mechanism, and a single past frame identifies it even after
current-centering removes the first shortcut. What survives both is deciding
whether an obstruction will clear, which requires deferring commitment under
temporal evidence. Two of the five conditions in `LearnedRecovery-v4` are
therefore load-bearing and three are not.

This is a more useful contribution than the original framing and does not
depend on the external Peg gate passing. Peg's value under this framing is to
test whether the same shortcut structure recurs in a task built on an official
ManiSkill base, and a negative answer there is publishable.

Threshold, seed family, and rejection history are unchanged. Every result in
this section is development evidence on an already-opened family.

## External validity of the audit itself

The ladder has been run on three benchmarks with an identical rung set. The
matched set revised an earlier conclusion recorded in this file and in the
README, and the revision is kept rather than absorbed.

| Benchmark | Best non-recurrent rung | Recurrent | Ratio | Verdict |
|---|---:|---:|---:|---|
| LearnedRecovery-v4 (ours) | 1.0000 | 1.0000 | 1.000 | shortcut |
| PegInsertionSide-v1 | 0.0909 | 0.4015 | 0.226 | none |
| REBOOT (external, real robot) | 0.7466 | 0.8045 | 0.928 | shortcut, marginal |

Two of three benchmarks, including real-robot trajectories collected by another
group, have held-out mechanisms identifiable without the capability under test.
This is a broader claim than the earlier "we audited our own benchmark and
found it flawed", and it does not depend on the Peg closed-loop gate passing.

PegInsertion is the discriminating negative that keeps the audit from being
vacuous: its strongest non-recurrent control reaches 0.226 of the recurrent
score, and its order-free summary is weaker than its two-frame control. The
separation between 0.226 and 0.928/1.000 is wide.

Three limitations attach and are not resolved.

The 0.9 line is a reporting convention set by this project, not a test with an
error rate. REBOOT at 0.928 is close to it and should be quoted as a ratio.

Every REBOOT figure rests on three optimizer seeds whose unstructured-GRU
spread is 0.7754 to 0.8506, wider than several of the differences discussed.
More seeds are needed before those intervals carry weight.

On REBOOT the factorized model does not improve on the capacity-matched
unstructured GRU (-0.0068, [-0.0249, 0.0119]). External evidence for the
architecture is absent rather than supportive, and on PegInsertion, the one
benchmark without a shortcut, it reaches 0.0199 on genuinely observed held-out
prefixes. The method claim is weak; the audit claim is what this work supports.
