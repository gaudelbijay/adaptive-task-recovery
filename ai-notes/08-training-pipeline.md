---
title: Training Pipeline
status: draft
last_updated: 2026-07-24
---

# Training Pipeline

## 1. End-to-end stages

```
Stage 0: Baseline task policies
  → train/import a competent policy per task (PickPlaceRecovery, DoorOpenRecovery,
    CarryWalkRecovery, PushRecoveryStand) with NO failure injection, NO recovery layer.
  → gate: task succeeds at high rate (>90%, task-dependent) under nominal conditions.

Stage 1: Failure injection curriculum design
  → sweep failure types/severities against the Stage-0 policies to find severity ranges
    that are "interesting" (cause failure often enough to matter, not so often the task
    policy never makes progress).
  → gate: documented severity ranges per failure type per task.

Stage 2: Failure detector training
  → generate labeled trajectories (nominal + injected-failure) from Stage-0 policies
    running under Stage-1 injection schedules.
  → train threshold baseline → ensemble-dynamics → sequence model, per 06-failure-taxonomy-and-detection.md.
  → gate: detector beats threshold baseline on precision/recall for ≥2 failure types.

Stage 3: Recovery skill training
  → for each learned skill (regrasp, re-approach, replan-and-retry): RL training with
    curriculum, using Stage-0 policy's own success check as the "resume" bonus target.
  → for classical skills (step-recovery): implement/tune the MPC/capture-point controller,
    validate against injected pushes.
  → gate: per-skill success rate under its target severity range.

Stage 4: Integrated system evaluation
  → wire arbiter + detector + skill library + task policy together end-to-end.
  → run full evaluation suite from 10-evaluation-and-benchmarks.md.
  → gate: end-to-end recovery success rate beats no-recovery and scripted-recovery baselines.

Stage 5 (stretch): Sim-to-real
  → see 09-sim-to-real-transfer.md.
```

Do not skip Stage 0's gate — a shaky baseline task policy makes every later measurement ambiguous (is the recovery policy bad, or was the task policy already unreliable?). Budget real time for this; it's the least glamorous stage but the one everything else's validity depends on.

## 2. Compute plan

- **Local GPU**: ManiSkill3's GPU-vectorized envs mean a single consumer GPU (e.g., RTX 4090) can likely provide meaningful parallel throughput for state-based observations; measure actual steps/sec early (see [04](04-simulation-environment-maniskill.md) §6) rather than assuming a number, and size batch-env count and training run length to fit your actual hardware/budget.
- **Cloud burst** (optional): if local compute is a bottleneck for the vision-based or larger-scale training runs, a short cloud GPU rental for specific experiments is more cost-effective than provisioning for peak load throughout — decide per-stage, not up front.
- **Track cost/throughput**, not just results, in [13-experiment-log-template.md](13-experiment-log-template.md) — "trained in N hours on 1 GPU" is a concrete, verifiable claim that's useful in interviews, unlike vague performance claims.

## 3. Logging / experiment tracking

- Use **Weights & Biases** (or TensorBoard if you want to stay fully offline/local) from the very first Stage-0 run — retrofitting logging after you already have "interesting" results is a common time-sink; set it up once, early.
- Log per-run: full config (see below), git commit hash, random seed, environment version/failure-injection schedule, and all metrics from [10-evaluation-and-benchmarks.md](10-evaluation-and-benchmarks.md) — not just training reward curves. Training reward alone does not tell you detection precision/recall or recovery success rate.
- Save periodic checkpoints with enough metadata (config + step count) to resume or re-evaluate any checkpoint later without guessing what produced it.

## 4. Configuration management

- Prefer **Hydra** (or a lighter hand-rolled YAML + dataclass loader if you want fewer dependencies) so every experiment is a versioned config file, not a pile of CLI flags remembered only in shell history.
- Directory convention: `src/atr/configs/{task}/{stage}/{experiment_name}.yaml`, with inheritance/overrides for sweeps (e.g., severity curriculum sweep, detector architecture sweep).
- Every config used for a reported number in the final writeup should be committed to the repo, not just logged to an external dashboard — reproducibility from the repo alone is a portfolio strength.

## 5. Reproducibility checklist

- [ ] Seeds set and logged for env, policy init, and any stochastic evaluation.
- [ ] `Dockerfile`/devcontainer pinning simulator, PyTorch, and CUDA versions — humanoid sim stacks are notoriously version-sensitive; don't rely on "works on my machine."
- [ ] `requirements.txt`/lockfile committed, not just a loose list of packages.
- [ ] A `scripts/reproduce_headline_result.sh` (or equivalent) that runs the exact pipeline producing your top-line reported number from a clean checkout — this is one of the highest-leverage single files for portfolio credibility; a reviewer who can actually reproduce your number believes the rest of the writeup far more.
