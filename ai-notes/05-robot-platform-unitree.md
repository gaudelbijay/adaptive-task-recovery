---
title: Robot Platform — Unitree Humanoids
status: draft
last_updated: 2026-07-24
---

# Robot Platform — Unitree Humanoids

> **Verify all numbers in this file against Unitree's current official datasheets before relying on them for engineering decisions (torque limits, control frequency, etc.).** Humanoid platform specs change across hardware revisions and firmware versions; treat everything quantitative below as "approximate, last known, re-check before use," not settled fact.

## 1. G1 vs H1 — why G1 is the default target

| Aspect | Unitree G1 | Unitree H1 |
|---|---|---|
| Size class | Smaller, ~human-child to small-adult scale | Full adult human scale, taller/heavier |
| Cost/accessibility | Lower cost, positioned as a more accessible research/dev platform | Higher cost, larger footprint, more powerplant to manage |
| DoF | High DoF including dexterous hand options in some configs | High DoF, historically arms less dexterous than G1's hand options in some SKUs |
| Use case fit | Tabletop manipulation + standing balance recovery — matches this project's v1 scope well | Better fit if the project scope grows into full-scale dynamic locomotion recovery |

**Decision:** default to **G1** for cost, footprint, and safety-during-testing reasons (a smaller/lighter robot falling is a smaller incident than a full-size H1 falling). Revisit if your actual hardware access is H1 — the software architecture in [03](03-system-architecture.md) doesn't change, only the URDF, torque limits, and physical safety protocol below.

## 2. Sensing and control interface

- **Proprioception**: joint encoders (position, often velocity and estimated torque), IMU (orientation, angular velocity, linear acceleration) at the base/torso.
- **Exteroception**: depth/RGB-D camera (head-mounted; exact model varies by SKU/generation — check current hardware config), optionally additional cameras or a LiDAR depending on configuration.
- **Control access**: Unitree provides an SDK (their `unitree_sdk2`-family C++/Python interfaces and, for some platforms, ROS2 bindings) for low-level joint command streaming and reading sensor state. Confirm the exact SDK version and API compatible with your specific unit/firmware before writing the `RobotInterface` real backend from [03](03-system-architecture.md) §3.
- **Control frequency**: real humanoid control loops typically run at a few hundred Hz to 1kHz at the low level (joint torque/PD loop), with higher-level policy inference running slower (tens of Hz) and the whole-body controller bridging the two — measure your actual achievable loop rate rather than assuming a number.

## 3. Safety protocol for real-world testing (mandatory before any powered-on test with a person nearby)

This project intentionally treats hardware phases as **safety-gated and staged** — do not skip steps to save time; a humanoid fall is a real injury/damage risk.

1. **Gantry/harness first**: initial balance-recovery tests should be done with the robot suspended or tethered by a fall-arrest harness/gantry so a bad recovery attempt cannot result in an uncontrolled fall.
2. **Hardware/software E-stop always within reach**: verify the E-stop actually cuts power/commands before starting *any* powered test, every session, not just once.
3. **Torque/velocity limits set conservatively below datasheet max** for all early testing; raise only after a specific behavior is validated at low limits.
4. **Shadow mode before enabling actions**: run the recovery policy "open loop observing" against the real robot — log what it *would* command — without actually sending commands, and inspect for anything alarming before ever closing the loop on hardware.
5. **Clear the area / minimum safe distance** for anyone not actively operating the E-stop.
6. **Staged capability rollout**: tabletop manipulation with the robot seated/supported → supported standing balance recovery (harnessed) → free-standing balance recovery (harnessed, loose tether) → free-standing, untethered — only after the previous stage has a clean track record. See [09-sim-to-real-transfer.md](09-sim-to-real-transfer.md) for the full staged deployment plan.
7. **Session logging**: every hardware session gets an entry in [13-experiment-log-template.md](13-experiment-log-template.md) including any near-misses, not just successes — near-miss logs are exactly what real robotics teams expect to see and are a credibility signal in a portfolio if shared appropriately.

## 4. Realistic expectations

Hardware access, cost, and safety overhead for a solo/small project mean the **real-robot phase is explicitly a stretch goal**, not the deliverable the whole project depends on. The simulation-only version of this project (Phases 0–4 in [11-roadmap-and-milestones.md](11-roadmap-and-milestones.md)) is already a complete, defensible, portfolio-worthy piece of work — frame the real-robot work as "extended it toward hardware" rather than treating simulation results as incomplete without it.

## 5. Open questions to resolve once hardware access is confirmed

- Exact G1 (or H1) SKU/configuration you'll have access to (hand end-effector type matters a lot for the manipulation tasks).
- Available compute onboard vs. offboard (will policy inference run on an onboard computer or streamed from a workstation over network — this affects your latency budget and therefore your domain-randomization latency range in [04](04-simulation-environment-maniskill.md) §5).
- Institutional/lab safety requirements if this is done anywhere other than a fully private space (insurance, supervision requirements, etc.).
