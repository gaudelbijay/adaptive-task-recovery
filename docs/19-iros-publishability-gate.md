---
title: IROS Publishability Gate and Method Reset
status: active-experiment
last_updated: 2026-08-30
---

# IROS publishability gate

The existing V19 result is a strong custom-benchmark result, not yet a strong
main-track IROS paper. Repository scale and the number of attempted variants do
not substitute for novelty or external validity. V36--V60 is closed as a
patching line after the independent V60 lineage retained only 69.53% nominal
safe success on seed 9351.

## Blocking weaknesses

1. The benchmark contains one intervention mechanism, two colored cubes, one
   camera, and one robot arm.
2. V19 is a privileged dual-teacher distillation recipe rather than a concise
   new recovery algorithm.
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
5. Oracle-belief, end-to-end V19, no-memory, random-feature, and
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

The assessment above predates the completed A+ V3 experiment. V3 fixed the
input mismatch and produced a positive causal-composition result: the causal
model reached 100% held-out reverse accuracy offline while the static and
unstructured controls reached 3.22% and 0%; on the once-only closed-loop
confirmation it achieved 573/576 held-out reverse safe successes and 2655/2880
overall, versus 2369/2880 for the matched unstructured baseline. The overall
gain was 9.93 points with a 95% Newcombe interval of [7.54, 12.29] points.

V3 is still rejected. Its nominal condition reached only 456/576 (79.17%),
below the frozen 82% worst-condition floor. This isolates a shared low-level
controller weakness rather than a router or causal-composition failure. V4
therefore changes only the nominal/post-clearance controller, identically for
all baselines, and reserves a fresh confirmation family. It does not reopen or
reinterpret V3.

Passing V4 is necessary but no longer sufficient for an A/A+ claim. The
remaining blocker is closed-loop external validity. The second-family protocol
is frozen separately in
`configs/a_plus_external_peg_insertion_gate_v1.json`: an intervention extension
of ManiSkill3's official `PegInsertionSide-v1`, chosen because contact-rich
six-DoF insertion is physically distinct from planar cube transport. Runtime
pose assignment and the older teleport-based TidyUp controllers are forbidden.
The external gate uses matched observations, shared specialists, a held-out
ejection direction, three seeds, a static shortcut ceiling, and a once-only
confirmation bank.

The release decision is conjunctive:

1. V4 primary gate passes on `333000000`.
2. The external PegInsertion gate passes on `429000000`.
3. The existing REBOOT result remains labeled offline real-robot transfer, not
   closed-loop real-robot control.
4. Costs and all rejected V-series candidates are compressed into auditable
   tables rather than presented as independent discoveries.

Until both closed-loop gates pass, the defensible paper framing is benchmark
and representation evidence, not a top-tier general recovery method.
