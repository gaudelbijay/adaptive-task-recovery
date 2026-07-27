---
title: Training Pipeline
status: draft
last_updated: 2026-07-26
---

# Training Pipeline

## Contributors and handoff contract

Person A owns the `ObservationWindow + instruction -> AgentBelief` path. Person B
owns the `AgentBelief + available skills -> guarded SkillCall` path. Both own the
schemas, benchmark, oracle, integration tests, and end-to-end evaluation. Model
selection is a measured task: downstream utility, calibration, generalization,
latency, memory, licensing, and integration cost must be recorded.

## Stage 0 — environment and oracle

**Shared:** implement one task family, goal schema, deterministic interventions,
constraint checker, and oracle feasibility planner. **Person B:** validate the
humanoid asset and navigation, reach, grasp, place, inspect, and safe-stop skills.
**Person A:** define trajectory capture and the placeholder belief adapter. Gate:
hand-authored tests and replayable episodes agree with oracle labels, and
low-level skill outcomes are logged separately.

## Stage 1 — static baseline

**Person B:** train static and oracle-feasibility language-conditioned policies
and evaluate both before and after interventions. **Person A:** supply the
deterministic language representation used by these baselines. **Shared gate:**
reliable nominal behavior and a measured adaptation gap.

Use pretrained/scripted low-level humanoid controllers where possible. Training
whole-body locomotion from scratch is a separate engineering track and must not
block validating the high-level research pipeline.

## Stage 2 — unlabeled visual data

**Shared:** freeze collection and evaluation splits. **Person A:** collect or
consume diverse unlabeled observation sequences and pretrain image, temporal,
and optionally object-centric visual encoders. **Person B:** benchmark inference
inside the policy loop and maintain compatibility with oracle beliefs.

## Stage 3 — feasibility model

**Shared:** generate labeled episodes using interventions and the oracle.
**Person A:** train per-goal feasibility and uncertainty heads. **Person B:** run
the learned beliefs through the fixed oracle-policy scaffold. Gate: beat simple
pixel-difference, supervised-feature, and majority baselines on held-out
compositions and demonstrate downstream value over matched noisy beliefs.

## Stage 4 — adaptive policy

**Person B:** train the high-level policy using learned feasibility beliefs and
fixed low-level skills, add the intent guard, and compare guard versus
reward-only constraints. **Person A:** support frozen versus jointly fine-tuned
representations and monitor calibration drift. **Shared:** own interface changes
and cross-module failure analysis.

## Stage 5 — end-to-end evaluation

**Person A:** lead representation, language, feasibility, calibration, and
counterfactual analyses. **Person B:** lead policy, guard, oracle-skill, and
humanoid execution analyses. **Shared:** run held-out splits and multiple seeds,
integrate results, and approve claims. Report failures, not just means.

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
