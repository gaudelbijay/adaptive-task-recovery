---
title: Training Pipeline
status: draft
last_updated: 2026-07-26
---

# Training Pipeline

## Stage 0 — environment and oracle

Implement one task family, goal schema, deterministic interventions, constraint
checker, and oracle feasibility planner. Validate a simulated humanoid asset and
at least the navigation, reach, grasp, place, inspect, and safe-stop skill
interfaces. Gate: hand-authored tests and replayable episodes agree with oracle
labels, and low-level skill outcomes are logged separately.

## Stage 1 — static baseline

Train a language-conditioned task policy in unchanged environments. Evaluate it
both before and after interventions. Gate: reliable nominal behavior and a
measured adaptation gap.

Use pretrained/scripted low-level humanoid controllers where possible. Training
whole-body locomotion from scratch is a separate engineering track and must not
block validating the high-level research pipeline.

## Stage 2 — unlabeled visual data

Collect diverse observation sequences without feasibility labels. Pretrain
self-supervised visual encoders using image, temporal, and optionally
object-centric objectives. Freeze collection splits before downstream training.

## Stage 3 — feasibility model

Generate labeled episodes using interventions and the oracle. Train per-goal
feasibility and uncertainty heads. Gate: beat simple pixel-difference,
supervised-feature, and majority baselines on held-out compositions.

## Stage 4 — adaptive policy

Train the high-level policy using learned feasibility beliefs, initially with
fixed low-level skills. Add the intent guard. Compare frozen versus jointly
fine-tuned representations and guard versus reward-only constraints.

## Stage 5 — end-to-end evaluation

Run all baselines, ablations, held-out splits, multiple seeds, calibration
analysis, and counterfactual tests. Report failures, not just means.

## Data discipline

- Version task generators, instruction grammars, intervention manifests, and splits.
- Separate unlabeled pretraining, downstream training, validation, and test seeds.
- Store simulator privileged state only in label/evaluation channels.
- Deduplicate near-identical trajectories across splits.
- Record model provenance and licenses for pretrained encoders and language models.

## Reproducibility

Every run records configuration, commit, seed, dependency lock, hardware,
dataset/split version, representation checkpoint, reward specification, and all
evaluation metrics. Use at least three seeds for development comparisons and
more for final claims when variance warrants it.

## Compute strategy

Start with low-dimensional actions, small images, short horizons, and frozen
encoders. Profile data loading and representation inference before scaling RL.
Do not choose a large vision-language backbone until a cheap oracle-state
pipeline demonstrates that the benchmark and evaluation can answer the question.
