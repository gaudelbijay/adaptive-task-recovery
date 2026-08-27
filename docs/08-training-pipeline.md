---
title: Training Pipeline
status: draft
last_updated: 2026-07-26
---

# Training Pipeline

## Scope areas and handoff contract

The representation area owns the `ObservationWindow + instruction ->
AgentBelief` path. The policy area owns the `AgentBelief + available
skills -> guarded SkillCall` path. Both areas own the schemas, benchmark,
oracle, integration tests, and end-to-end evaluation. Model selection is
a measured task: downstream utility, calibration, generalization,
latency, memory, licensing, and integration cost must be recorded.

## Stage 0 — environment and oracle

**Shared:** implement one task family, goal schema, deterministic interventions,
constraint checker, and oracle feasibility planner. **Policy:** validate the
humanoid asset and navigation, reach, grasp, place, inspect, and safe-stop skills.
**Representation:** define trajectory capture and the placeholder belief adapter. Gate:
hand-authored tests and replayable episodes agree with oracle labels, and
low-level skill outcomes are logged separately.

## Stage 1 — static baseline

**Policy:** train static and oracle-feasibility language-conditioned policies
and evaluate both before and after interventions. **Representation:** supply the
deterministic language representation used by these baselines. **Shared gate:**
reliable nominal behavior and a measured adaptation gap.

Use pretrained/scripted low-level humanoid controllers where possible. Training
whole-body locomotion from scratch is a separate engineering track and must not
block validating the high-level research pipeline.

## Stage 2 — unlabeled visual data

**Shared:** freeze collection and evaluation splits. **Representation:** collect or
consume diverse unlabeled observation sequences and pretrain image, temporal,
and optionally object-centric visual encoders. **Policy:** benchmark inference
inside the policy loop and maintain compatibility with oracle beliefs.

## Stage 3 — feasibility model

**Shared:** generate labeled episodes using interventions and the oracle.
**Representation:** train per-goal feasibility and uncertainty heads. **Policy:** run
the learned beliefs through the fixed oracle-policy scaffold. Gate: beat simple
pixel-difference, supervised-feature, and majority baselines on held-out
compositions and demonstrate downstream value over matched noisy beliefs.

## Stage 4 — adaptive policy

**Policy:** train the high-level policy using learned feasibility beliefs and
fixed low-level skills, add the intent guard, and compare guard versus
reward-only constraints. **Representation:** support frozen versus jointly fine-tuned
representations and monitor calibration drift. **Shared:** own interface changes
and cross-module failure analysis.

## Stage 5 — end-to-end evaluation

**Representation:** lead representation, language, feasibility, calibration, and
counterfactual analyses. **Policy:** lead policy, guard, oracle-skill, and
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

## Non-teleport manipulation track

The high-level TidyUp Q/BC experiments use the established abstract skill
executor and must be labeled decision-layer diagnostics. Physical manipulation
is trained separately with `scripts/train_manipulation_ppo.py`; that trainer
imports ManiSkill tasks directly and does not import an ATR teleport executor.
`configs/manipulation_ppo_v1.json` matches ManiSkill v3.0.0b22's official
three-seed state-PPO baseline settings for PickCube, randomized PickSingleYCB,
and Unitree G1 apple-in-bowl (50M transitions per seed/task).
The 1,024-environment G1 task explicitly reserves a 256 MiB PhysX collision
stack. A capacity warning invalidates the affected run even if the process
continues; partial 4 MiB, 16 MiB, and 64 MiB runs are quarantined rather than
resumed. The 64 MiB setting was itself rejected after one seed requested
69,717,648 bytes at about 11.4M transitions; all seeds use the same corrected
capacity rather than mixing simulator configurations.

Each task writes atomic `latest.pt` and `best.pt` checkpoints containing model,
optimizer, iteration/global-step counters, and Python/NumPy/Torch/CUDA RNG
state. Jarvis sends `SIGUSR1` five minutes before the 24-hour limit; the trainer
saves at the next iteration boundary. Submit an `afterany` continuation array
with the identical immutable config. Simulator state itself is not portable
across jobs, so a continuation re-seeds the environment stream while resuming
the full optimization state; report that limitation.

Final evidence comes from `evaluate_manipulation_ppo.py`, not the
checkpoint-selection evaluations embedded in training. It loads `best.pt`,
uses a disjoint seed range and reconfiguration on every vector slot, runs 256
deterministic held-out episodes per training seed, and records Wilson intervals.
`aggregate_manipulation_results.py` refuses to aggregate until all nine held-out
artifacts exist.

## Integrated learned-control recovery

`configs/learned_recovery_ppo_v6.json` is the frozen experiment that removes
the hierarchy between ATR's recovery decision and its motor controller. One
Panda PPO policy receives the factorized two-goal instruction, continuous robot
and scene state, and goal-progress memory. During the same episode, a dynamic
sweeper physically removes either requested cube; the policy must resolve the
ordered feasible suffix while keeping a protected object fixed. Actor pose
assignment is confined to randomized reset. The intervention uses applied
force and contact dynamics, and policy execution contains no teleport path.

The matched three-seed comparison trains 100M requested transitions per seed:

- adaptive PPO, trained with a 50/50 mix of nominal and intervention episodes;
- privileged-oracle PPO, with the same training distribution plus explicit
  unavailable-goal bits;
- no-intervention-training PPO, trained only in nominal worlds and evaluated
  under the same intervention as the other policies.

All methods share PPO architecture, optimizer settings, continuous action
space, safety shaping, and checkpoint budget. A protected-object displacement
terminates an episode. Best checkpoints maximize validation success minus two
times the validation failure rate, with return used only as a tiny tie-break.
The Slurm script atomically saves model, optimizer, counters, and RNG state on
`SIGUSR1` and automatically resubmits an incomplete run before Jarvis's 24-hour
limit. No simulator state is claimed to survive a job boundary.

```bash
mkdir -p results/slurm
ATR_PYTHON=.venv/bin/python \
  sbatch --array=0-8%9 scripts/slurm_manipulation_ppo.sh

# Submit again with --dependency=afterany:<training-job-id> for 24 h resume.
# Submit evaluation after the continuation succeeds.
sbatch --array=0-8%9 --dependency=afterok:<continuation-job-id> \
  scripts/slurm_manipulation_eval.sh
```
