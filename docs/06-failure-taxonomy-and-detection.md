---
title: Failure Taxonomy and Detection
status: draft
last_updated: 2026-07-24
---

# Failure Taxonomy and Detection

## 1. Failure taxonomy table

| Category | Example | Primary detectable signal(s) | Severity range | Ground truth in simulation |
|---|---|---|---|---|
| Perception | Occlusion, segmentation failure, pose estimate jump | vision confidence, temporal pose discontinuity | low–med | object pose vs. simulator ground truth |
| Contact / manipulation | Grasp slip, contact loss, misalignment | contact flags, object velocity vs. hand velocity | med–high | exact contact constraint state and object pose |
| Balance / locomotion | External push, uneven terrain, trip | IMU tilt angle/rate, CoM-vs-support-polygon margin, foot contact timing | high | exact CoM, support polygon, and contact forces |
| Planning / state | Object moved, stale world model, drawer already open | discrepancy between expected and observed state at sub-goal checkpoints | low–med | full state diff vs. plan's assumed state |
| Actuation | Torque saturation, motor fault, backlash | commanded vs. achieved joint torque/position tracking error | med–high | exact simulated motor-model deviation |
| Whole-body coupling | Manipulation destabilizes stance | CoM shift correlated with arm motion exceeding a learned/derived margin | med–high | full-body simulator state |

Severity should be treated as a **continuous parameter** in the failure-injection API ([04](04-simulation-environment-maniskill.md) §4), not just these coarse bins — use the bins for reporting/aggregation, not for the underlying injection code.

## 2. Detection approaches, in increasing sophistication (build in this order)

1. **Threshold baseline** (must build first — this is your comparison point for every later claim): fixed or lightly-tuned thresholds on IMU tilt, tracking error, contact-force deviation. Cheap, fast, zero training, and a legitimate real system many companies actually ship — beating it convincingly is the first real result of the project.
2. **Ensemble dynamics-disagreement**: train k forward-dynamics models (predict s_{t+1} from s_t, a_t) on nominal trajectories; at test time, high variance across the ensemble's predictions flags an anomaly. Naturally handles "situations unlike training" without needing failure labels.
3. **Sequence anomaly model**: a small Transformer/LSTM over a sliding window of proprioceptive (+ optionally visual-feature) history, trained either (a) unsupervised as a next-step predictor with reconstruction/prediction error as the anomaly score, or (b) supervised as a classifier if you have labeled failure windows from the injection API. Do both if time allows — comparing supervised vs. unsupervised detection is a strong ablation for the writeup.
4. **Multi-modal fusion**: combine proprioceptive and visual anomaly scores (e.g., late fusion of two scores, or a joint model) — only worth doing after single-modality baselines are solid and measured separately, so you can report the fusion's actual marginal benefit.

## 3. Labeling strategy

- **In simulation**: the failure-injection API gives you exact onset time and type for every injected failure — use this as ground truth for supervised training and for computing precision/recall/latency. Also log *naturally occurring* failures (task policy fails on its own, e.g. missed grasp with no injection) separately, since these are your best test of generalization beyond the injected-failure distribution.

## 4. Metrics

- **Precision / recall / F1** of failure detection against injected ground truth, computed per failure type and severity, not just aggregated — an aggregate number hides whether you're good at detecting slips but blind to occlusion, which matters for both engineering and interview discussions.
- **Detection latency**: time (steps or ms) between injected failure onset and the monitor crossing its decision threshold. This trades off against precision (lower threshold = faster but noisier) — report the full precision/recall-vs-latency curve, not a single operating point.
- **False-positive rate during nominal execution**: how often the monitor fires when nothing is actually wrong — critical, since a jumpy detector that constantly interrupts the task policy is worse than no detector at all.
- **Generalization gap**: detector performance on failure types/severities present in training vs. held out entirely — this is the number that actually answers RQ1/RQ4 from [01](01-problem-statement-and-motivation.md).

## 5. Practical build order

1. Threshold baseline on `PushRecoveryStand` (simplest env) using IMU tilt only.
2. Extend threshold baseline to all four v1 environments and all implemented failure types — get an end-to-end precision/recall table before touching a learned model.
3. Ensemble-dynamics detector, same evaluation.
4. Sequence anomaly model, same evaluation.
5. Only then: multi-modal fusion and any architecture tuning — don't optimize the learned detector's architecture before the evaluation harness itself is trustworthy.
