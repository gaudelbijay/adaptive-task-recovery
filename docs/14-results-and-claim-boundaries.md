---
title: Results and Claim Boundaries
status: active
last_updated: 2026-08-27
---

# Results and claim boundaries

This is the paper-facing index of results that passed their current validation
gate. It deliberately separates abstract skill selection from continuous robot
control. Confidence intervals are 95%; a point interval means the aggregate was
constant across the evaluated high-level split, not that uncertainty is absent.

## Frozen cross-embodiment adaptation benchmark

The v1 run contains 3,200 paired cases and 12,800 policy episodes across four
environment families. Oracle feasibility and static execution both achieve
1.68625 goals/case. Static execution uses 14.24 more wasted steps per paired
case (paired bootstrap interval 12.708--15.842). This supports a recovery
efficiency claim, not a higher-goal-recall claim.

The v1 safety columns are invalid because the evaluator originally read an
optional policy-specific field. The immutable v1 records remain preserved; no
safety result is taken from them.

## Corrected effect-aware safety benchmark

The v3 run contains 500 paired humanoid cases and 2,000 policy episodes. Every
policy is scored by the same final environment oracle.

| Policy | Goals/case | Wasted steps/case | Violations/case |
|---|---:|---:|---:|
| Static | 1.690 [1.650, 1.730] | 7.750 [6.750, 8.750] | 1.000 [1.000, 1.000] |
| Oracle feasibility | 1.690 [1.650, 1.730] | 1.550 [1.050, 2.100] | 0.752 [0.714, 0.790] |
| Unguarded substitution | 1.682 [1.642, 1.722] | 7.950 [6.950, 8.950] | 0.778 [0.742, 0.814] |
| Effect-aware guard | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

The result exposes a safety/recall frontier: the guard blocks all observed
violations, but the current fixed bowl skill is predicted to move a protected
glass and there is no safe alternative skill. Feasibility alone is therefore
not a safety mechanism.

## Learned high-level policy diagnostic

These policies act through the benchmark's teleport-on-success executor. They
test decisions under held-out intervention mechanisms and must not be described
as learned physical manipulation.

| Policy | Training seeds | Test episodes | Goals | Wasted steps |
|---|---:|---:|---:|---:|
| Feasibility Q | 10 | 640 | 1.375 | 6.640625 |
| Behavioral cloning | 10 | 640 | 1.375 | 6.640625 |
| Privileged mechanism-ID Q | 10 | 640 | 0.000 | 0.000 |
| Blind domain-randomized Q | 10 | 640 | 1.000 | 0.000 |
| Scripted oracle feasibility | 0 | 64 | 1.375 | 6.640625 |
| Scripted static | 0 | 64 | 1.375 | 15.625 |

Feasibility Q and behavioral cloning match the scripted oracle means on this
split. The privileged mechanism-ID policy collapses on unseen identities,
while the blind policy sacrifices recall to avoid waste. This is evidence
about representation and reward behavior, not evidence of general robot skill.

## Continuous non-teleport manipulation

Three-seed, 50M-transition PPO runs use ManiSkill's official task-specific
parallelization settings for PickCube-v1, randomized PickSingleYCB-v1, and
UnitreeG1PlaceAppleInBowl-v1. Each result below uses the best checkpoint chosen
only by training-stream evaluations, then a deterministic evaluation on 256
episodes per seed whose seeds are disjoint from training and checkpoint
selection. The pooled interval is a 95% Wilson interval over 768 held-out
episodes; seed dispersion is sample standard deviation across three rates.

| Task | Held-out successes | Pooled success | Seed mean ± SD |
|---|---:|---:|---:|
| PickCube-v1 | 755/768 | 98.31% [97.13%, 99.01%] | 98.31% ± 0.90% |
| PickSingleYCB-v1 | 530/768 | 69.01% [65.65%, 72.18%] | 69.01% ± 4.30% |
| UnitreeG1PlaceAppleInBowl-v1 | 767/768 | 99.87% [99.27%, 99.98%] | 99.87% ± 0.23% |

All nine training jobs reached their batch-aligned budgets (49,954,816 for
PickCube, 49,938,432 for PickSingleYCB, and 49,971,200 for G1), all nine
checkpoint-continuation checks exited successfully, and all nine held-out
evaluations reported every requested success trial. The G1 runs use an
explicit 256 MiB PhysX collision stack; their complete native logs contain
zero overflow diagnostics. Partial 4 MiB, 16 MiB, and 64 MiB G1 runs are
quarantined and excluded. The full repository validation is 345 passed and
zero failed across four deterministic, disjoint test-file shards.

These standard-task results establish learned continuous manipulation without
ATR's teleport executor. They do not establish adaptation after irreversible
changes, language conditioning, visual-policy learning, or transfer to the
Fetch ReplicaCAD task. Exact numerical ranking against ManiSkill's published
curves is also avoided because this run uses the readable PPO implementation,
a separately declared held-out protocol, and no CUDA-graph optimization.

The separate Fetch ReplicaCAD controller physically grasps, carries, and places
the can in 10/10 sequential episodes. Its current scripted bowl grasp fails in
10/10, giving 1.0/2.0 mean goals, 0/10 complete tasks, and zero constraint
violations. A 69-candidate position/approach/torso sweep produced zero
contact-verified grasps and zero retained lifts. A separate 12-candidate 6-DoF
diagnostic could not reach its requested orientations and is not counted as a
valid grasp test. The position sweep rejects the tested scripted grasp family;
it is not a proof that the Fetch embodiment can never grasp the bowl. Standard
ManiSkill PPO success does not imply transfer to that Fetch scene or solve the
full ATR physical task.

## Comparison rule

Related systems use different embodiments and task definitions. Compare ATR to
SayCan, Inner Monologue, KnowNo, and shielding by capability and assumptions as
documented in [02-background-and-related-work.md](02-background-and-related-work.md).
Use numerical ranking only for policies evaluated on the same ATR cases or the
same held-out ManiSkill protocol.
