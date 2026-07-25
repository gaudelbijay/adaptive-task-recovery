---
title: Sim-to-Real Transfer
status: draft
last_updated: 2026-07-24
---

# Sim-to-Real Transfer

This entire file is a **stretch-goal plan**, gated on confirmed hardware access (see [05-robot-platform-unitree.md](05-robot-platform-unitree.md) §4). The simulation-only project is complete and valuable without it.

## 1. Sources of sim-to-real gap, specific to this project

- **Physical parameter mismatch**: mass, friction, motor dynamics, joint damping/stiffness in sim vs. the real G1/H1.
- **Sensing mismatch**: real IMU/encoder noise characteristics, real camera latency/distortion/exposure vs. simulated sensors.
- **Control loop latency**: real command-to-effect latency (network, SDK, actuator response) vs. simulated instantaneous or fixed-delay actuation.
- **Contact modeling**: SAPIEN/PhysX contact dynamics (friction cone approximation, restitution) are an approximation of real contact physics — especially significant for the grasp-slip and step-recovery skills, which are literally *about* contact behavior at the margin.
- **Failure-injection realism**: your simulated "failures" (force impulses, friction drops) are a hypothesis about what real failures look like — real failures may have different signatures the detector never saw.

## 2. Domain randomization (extends [04](04-simulation-environment-maniskill.md) §5)

Widen randomization ranges specifically around the parameters most likely to be mis-specified: friction coefficients, joint damping/stiffness, sensor noise magnitude and latency, and camera intrinsics/extrinsics jitter if using vision. Validate that policies trained under randomization still solve the *nominal* task well in sim before ever testing on hardware — a policy that's become too conservative under randomization to do the task at all has not usefully improved sim2real, it's just failed differently.

## 3. System identification (real → sim calibration)

1. Collect short real-robot logs of simple, safe motions (e.g., single-joint sinusoids, standing sway) with the robot harnessed.
2. Compare real joint tracking response and IMU response against sim under the same commanded trajectory.
3. Adjust sim physical parameters (joint damping/stiffness, latency model, mass/inertia if available from datasheet) to reduce this gap — a lightweight manual or grid-search calibration is enough for v1; a fully automated real2sim optimization loop is a reasonable stretch/future-work item, not a v1 requirement.
4. Re-validate Stage-0 baseline task policies still perform acceptably in the recalibrated sim before re-running any downstream training — recalibrating sim physics can silently invalidate policies trained under the old parameters.

## 4. Safety wrapper for real deployment

- Hard torque/velocity limits enforced in software **below** datasheet max, independent of whatever the policy outputs — a clamp the policy cannot override.
- A watchdog that forces `abort-to-safe-pose` (or a full E-stop) if control loop timing, sensor freshness, or command bounds are violated, independent of the failure monitor itself (the failure monitor can be wrong; the watchdog is a dumb, trustworthy backstop).
- **Shadow mode**: run the full pipeline against live real-robot sensor input with actions logged but not sent, and manually review logs for anything that would have been unsafe, before ever closing the loop.
- Human operator with E-stop physically present for every powered test, full stop — no exceptions, no "just a quick check."

## 5. Staged deployment plan (mirrors [05](05-robot-platform-unitree.md) §3)

| Stage | Setup | Goal |
|---|---|---|
| A | Robot seated/mechanically supported | Validate manipulation-recovery skills (`regrasp`, `re-approach`) with zero balance risk |
| B | Standing, harnessed/gantry-supported | Validate `step-recovery` under mild injected pushes, harness bearing any fall |
| C | Standing, loose tether | Validate `step-recovery` under nominal pushes with tether as backup only |
| D | Free-standing, untethered (only after clean track record at C) | Full integrated system demo |

Do not advance a stage until the previous stage has run cleanly across multiple sessions — log every session, including near-misses, in [13-experiment-log-template.md](13-experiment-log-template.md).

## 6. What "success" looks like for the stretch goal

Given the realistic scope, a strong stretch-goal outcome is: **Stage A validated end-to-end on hardware, with Stage B attempted and honestly reported** (including failures/limitations) — this is a legitimate, impressive result and should be presented as such, rather than treating anything short of full free-standing autonomy as a shortfall. Reviewers and interviewers respond well to honest "here's how far I got and exactly what the remaining gap is" narratives; it's more credible than an overclaimed full success.
