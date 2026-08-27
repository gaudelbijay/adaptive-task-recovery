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

## Integrated non-teleport Fetch recovery

This is the first experiment that combines the ATR components in one physical
episode rather than evaluating decision and manipulation tracks separately. A
parsed instruction requests the potted-meat can and cracker box while protecting
the master-chef can. The cracker is irreversibly removed during real can
execution. A fixed-camera RGB change score supplies feasibility, a physically
trained Q table chooses attempt/skip, the intent/navigation guard screens the
action, and `attempt_goal_with_real_grasp()` performs real navigation, IK,
contact-verified grasp, carry, and release. The module never imports the
teleport executor.

| Policy | Achievable can completion | Wasted steps | Violations |
|---|---:|---:|---:|
| Static | 0.633 [0.467, 0.800] | 461.6 [370.4, 552.9] | 0.000 [0.000, 0.000] |
| Privileged oracle | 0.600 [0.433, 0.767] | 191.6 [111.8, 271.4] | 0.000 [0.000, 0.000] |
| Visual learned + guard | 0.767 [0.600, 0.900] | 111.8 [47.9, 191.6] | 0.000 [0.000, 0.000] |

The RGB classifier is correct in 30/30 intervention episodes per policy and
the 90-record audit sums to zero teleport calls. Visual learned minus static
wasted steps is -349.9 with paired-bootstrap interval [-470.8, -226.8]. Its
destroyed-goal-only waste is 0.0 versus static's 285.9 [274.6, 297.3], which
isolates adaptation from stochastic failures of the shared can controller. Its
completion difference is +0.133 with interval [-0.100, +0.367], so completion
improvement is not claimed. Physical can outcomes are simulator-stochastic
even under paired seeds; this makes the efficiency result stronger than a
raw comparison of unpaired success percentages.

This closes the integration gap only at a hierarchical level. Perception is a
scene/object-calibrated frame-difference detector, the learned component is a
two-action high-level Q table, and the low-level motor skill remains scripted.
The destroyed second object means the experiment measures valid partial
completion, not two-object physical success.

## Integrated learned continuous-control recovery

`LearnedRecovery-v1` removes the hierarchy between recovery selection and motor
execution. A Panda policy acts only through continuous `pd_joint_delta_pos`
commands. It receives a factorized two-goal order, goal-progress memory, and
simulator state. A dynamic sweeper physically removes one requested cube with
applied force during the episode; the policy must complete the remaining
ordered feasible goals without displacing a protected yellow object. Pose
assignment is confined to randomized reset, and a runtime regression test
forbids `Actor.set_pose()` during task execution.

V6 trains three matched PPO methods for exactly 99,942,400 transitions per
seed, three seeds each. Checkpoint selection uses only training-stream
validation and scores success minus twice the failure rate. Final evaluation
uses 256 disjoint held-out episodes per seed under intervention and another 256
per seed under the nominal condition. Methods share held-out seeds. The primary
endpoint is safe success: task success and no constraint violation at any time.

![V6 learning curves across all three seeds, with independent held-out raw and safe success overlays.](../media/results/learned-recovery-v6-curves.png)

The vector version is [available as PDF](../media/results/learned-recovery-v6-curves.pdf).

| Method | Raw held-out success | Violation | Safe held-out success | Safe seed mean ± SD |
|---|---:|---:|---:|---:|
| **Adaptive PPO** | 459/768, 59.77% [56.26, 63.18] | 8.59% | **397/768, 51.69% [48.16, 55.21]** | 51.69 ± 15.24% |
| Privileged unavailable-state PPO | 500/768, 65.10% [61.67, 68.39] | 20.83% | 354/768, 46.09% [42.60, 49.63] | 46.09 ± 6.81% |
| No-intervention-training PPO | 295/768, 38.41% [35.04, 41.90] | 4.95% | 279/768, 36.33% [33.00, 39.79] | 36.33 ± 15.61% |

Paired effects over the same 768 intervention episodes are:

| Comparison | Raw-success difference | Safe-success difference |
|---|---:|---:|
| Adaptive − no-intervention training | +21.35 [17.32, 25.39] | **+15.36 [10.68, 20.05]** |
| Adaptive − privileged unavailable state | −5.34 [−9.51, −1.17] | **+5.60 [1.17, 10.03]** |

The raw/safe reversal against the privileged policy is not described as
adaptive dominance: privileged state yields more raw completions, while its
selected policies move the protected object more often. The constrained metric
prefers adaptive PPO because the experiment defines those runs as failures.

Branch stratification rules out the V2 shortcut in which success concentrated
almost entirely on second-goal removal:

| Removed goal | Adaptive safe success | No-intervention safe success | Paired difference |
|---|---:|---:|---:|
| First requested goal (hard recovery) | 33.24% [28.56, 38.27] | 0.00% [0.00, 1.06] | **+33.24 [28.49, 38.27]** |
| Second requested goal | 67.80% [63.13, 72.15] | 68.05% [63.39, 72.37] | −0.24 [−7.56, 7.07] |

The adaptive gain is isolated to the branch that requires abandoning the
unavailable first goal, while safe performance is unchanged when the nominal
prefix remains valid. Nominal safe success is 33.46% for adaptive, 37.24% for
privileged, and 0% for no-intervention training; none of these policies should
be presented as a solved two-object controller.

This is the strongest same-task evidence for H2, but it is not the complete
vision-language hypothesis. Observations are low-dimensional simulator state,
language is the factorized order encoding rather than open-vocabulary text,
the object set and intervention mechanism are narrow, adaptive safe success is
only 51.69%, and seed dispersion is substantial. The 8.59% adaptive violation
rate also means reward shaping plus termination does not replace the explicit
runtime guard validated elsewhere under H3.

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
quarantined and excluded. The latest full repository validation is 348 passed and
zero failed across four deterministic, disjoint test-file shards.

These standard-task results establish learned continuous manipulation without
ATR's teleport executor. The separate integrated Fetch result above establishes
hierarchical adaptation, but these PPO checkpoints themselves do not establish
adaptation after irreversible changes, language conditioning, or transfer to
the Fetch ReplicaCAD task. Exact numerical ranking against ManiSkill's published
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
