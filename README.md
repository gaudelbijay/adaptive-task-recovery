# Adaptive Task Recovery

Adaptive Task Recovery (ATR) is a research project about **intent-preserving
adaptation after irreversible world changes**. It studies whether a
vision-language reinforcement learning agent can use self-supervised visual
representations to determine which language-specified goals remain feasible and
revise its strategy to achieve as much of the original intent as possible.

<p align="center">
  <img src="media/demos/learned-recovery-montage.gif" width="900" alt="Three frozen-policy ManiSkill recordings of one restricted-RGB Panda policy: recovery after the first requested cube is physically removed, recovery after the second cube is physically removed, and nominal completion of both ordered goals.">
</p>

<p align="center">
  <img src="media/demos/manipulation-task-montage.gif" width="680" alt="Four real ManiSkill recordings: a Fetch robot physically picks, carries, and places a can; a Panda arm picks a cube; a Panda arm picks a randomized YCB object; and a Unitree G1 humanoid places an apple in a bowl.">
</p>

<p align="center">
  <img src="media/demos/fetch-safety-detour.gif" width="360" alt="A Fetch robot screens its planned route, finds a protected object sitting on it, and replans a detour instead of pushing through.">
</p>

<p align="center"><sub>
Every panel is a real ManiSkill3 recording. The hero replays one frozen Panda
PPO policy across both irreversible-change orderings and a nominal two-goal
episode. The gallery retains physical Fetch pick/carry/place and the three
standard learned-control tasks; the final recording shows collision-aware
Fetch replanning. See <a href="#what-this-project-does-not-claim">what this
project does not claim</a> for the boundary between evaluated tracks.
</sub></p>

## Research question

A robot gets an instruction with more than one goal. Partway through, the
world changes in a way that can't be undone — something breaks, disappears,
or a path closes. Can the robot tell which goals are still possible, still
do whichever of them remain achievable, and never fake success by doing
something it was never asked to do?

> Can a vision-language reinforcement learning agent, equipped with
> self-supervised visual representations, learn to identify which
> language-specified goals remain feasible after unforeseen and irreversible
> world changes, and adapt its task strategy to maximize goal achievement
> without violating the original intent?

See [`docs/00-project-overview.md`](docs/00-project-overview.md) for the
full breakdown, including the build-up order this project actually follows
(language, then vision, then self-supervised representations, then a
learned policy — one capability at a time, each checked before the next).

Examples include an object becoming unavailable, a container breaking, a route
becoming blocked, or a limited resource being consumed. These changes make the
original plan invalid and may make some goals impossible. The challenge is to
distinguish infeasibility from temporary difficulty, reason over multiple goals
and constraints, and choose an acceptable partial or alternative completion.

## Results

Five hypotheses, each with real evidence — paired seeds, bootstrap confidence
intervals, and real ManiSkill3 simulator episodes, not toy numbers. Full
detail and every underlying number is in
[`ai-notes/decisions.md`](ai-notes/decisions.md); the validated result index
and claim boundaries are in
[`docs/14-results-and-claim-boundaries.md`](docs/14-results-and-claim-boundaries.md).

### 2026 causal-recovery audit: real trajectories expose the simulator shortcut

The latest candidate was evaluated on the May 2026
[REBOOT real-robot benchmark](https://nanayawoa.github.io/REBOOT/) before any
README promotion. The pinned audit contains **2,072 usable trajectories from
37 repositories**, evaluated by leaving out each of nine object families in
turn. Inputs and splits are matched across methods; repository SHAs, excluded
episodes, optimizer seeds, and every held-out fold are recorded in
[`results/a_plus_audit`](results/a_plus_audit/reboot_v2_aggregate.json).

| REBOOT recovery-state predictor | Leave-one-object-out macro-AUROC |
|---|---:|
| Static MLP (last observation) | 0.5797 |
| Trajectory-moment MLP | 0.7450 |
| Capacity-matched unstructured GRU | 0.8072 |
| **Causal dynamics GRU** | **0.8353** |

The structured causal model improves over the static model by **+25.56 AUROC
points** (object-bootstrap 95% CI **+21.10 to +29.47**) and over the strong
trajectory-moment baseline by **+9.03 points** (**+1.96 to +18.08**). It is
also +2.82 points above the unstructured GRU, but that narrower interval
crosses zero (**−1.40 to +9.38**), so the repository does not claim a
significant architecture win. This is an offline real-robot recovery-prefix
result, not real-robot closed-loop control.

The matched simulator audit reached **878/960 — 91.46% safe success** with
**16/960 — 1.67% violations**, but the unstructured recurrent router produced
the exact same closed-loop outcome and the static router was only one episode
behind (**877/960**). The frozen A+ gate therefore correctly rejects the local
method-superiority claim: recurrence is valuable on REBOOT, while the current
simulator exposes the mechanism in a single state and cannot establish that
claim. The failure and shortcut-probe artifacts are retained rather than
hidden; the previous V4 confirmatory headline below remains the released
closed-loop result.

| | Hypothesis | Result |
|---|---|---|
| **H1** | A perceptual feasibility signal (not just privileged simulator state) is usable | **Confirmed.** Real zero-shot CLIP perception matches oracle behavior exactly on the project's own success-criteria benchmark — same recall, real reduction in wasted steps. A robustness gap was found by actually running the benchmark, then fixed and re-verified, not assumed away. |
| **H2** | Feasibility-aware policies beat blindly continuing the plan | **Confirmed.** In the new continuous-control benchmark, adaptive PPO improves safety-qualified held-out success over no-intervention training by **+15.4 points** (paired 95% CI **+10.7 to +20.1**). On the hard first-goal-removed branch the gain is **+33.2 points** (**+28.5 to +38.3**) and is statistically unchanged on the easy branch. Earlier Fetch/apartment tests independently show lower wasted execution at matched recall. |
| **H3** | The safety guard does real work, not "safe by doing nothing" | **Confirmed.** Built the two adversarial cases meant to break this claim — a goal that directly conflicts with a protected object, and a case where the guard turned out to be *too permissive*. Both found real bugs, both fixed, both locked into regression tests. Extended to real collision-aware navigation with live detours and fail-closed stops, verified under seed variance, not one staged scenario. |
| **H4** | A factorized language representation generalizes; memorization doesn't | **Confirmed.** The real parser hits 100% across train, held-out paraphrase, and held-out composition splits. A hand-built monolithic memorizer hits 100% on train and 0% on anything held out. Re-verified on the full 180-case combinatorial sweep of the object pool, not a hand-picked sample. |
| **H5** | Calibrated abstention beats forced decisions | **Confirmed — conditionally.** Abstaining only wins when a wrong forced decision is the expensive mistake; tested both directions honestly, not just the flattering one, and confirmed with 30-seed bootstrap intervals that exclude zero in both directions (`[−0.20, −0.06]` where abstention wins, `[+0.10, +0.14]` where it loses). The unconditional version of this hypothesis was wrong; the conditional one is real. |

The separate non-teleport manipulation track trained three seeds per task for
50M requested transitions and evaluated 256 disjoint-seed episodes per
checkpoint. Intervals below are pooled 95% Wilson intervals.

| Continuous-control task | Held-out success |
|---|---:|
| PickCube-v1 | **755/768 — 98.31%** [97.13%, 99.01%] |
| PickSingleYCB-v1 | **530/768 — 69.01%** [65.65%, 72.18%] |
| UnitreeG1PlaceAppleInBowl-v1 | **767/768 — 99.87%** [99.27%, 99.98%] |

### Mechanism-diverse V4 recovery: frozen confirmatory result

The new `LearnedRecovery-v4` benchmark prevents the earlier result from being
explained by one familiar removal animation. At step 0, one of four
force-driven mechanisms occurs: forward ejection, reverse ejection, permanent
goal blockage, or visually identical temporary blockage. Nominal episodes are
included in the same evaluation. A 1,024-episode physics audit observed the
intended intervention in **1,024/1,024 episodes with zero collateral target
loss**.

The final hybrid controller detects motion from explicit object poses, waits
for causal evidence before treating a blockage as permanent, and hands control
to mechanism-specific continuous-control policies. The permanent specialist
is adapted under the exact 36-step handoff it receives at deployment; the
temporary branch resumes only after the blocker is physically clear. No hidden
mechanism ID or intervention-target label is read by the controller.

After all controller development was frozen, an untouched `280000000` seed
family was run for both methods (3 paired seed lineages × 5 conditions × 64
episodes = 960 episodes per method):

| V4 condition | Frozen hybrid success | V19 baseline success | Hybrid violations | V19 violations |
|---|---:|---:|---:|---:|
| Nominal | 160/192 — 83.33% | **172/192 — 89.58%** | 6.77% | 6.77% |
| Forward ejection | 167/192 — 86.98% | **186/192 — 96.88%** | **0%** | 2.60% |
| Permanent blockage | **187/192 — 97.40%** | 161/192 — 83.85% | **0%** | 14.06% |
| Temporary blockage | **168/192 — 87.50%** | 111/192 — 57.81% | **3.65%** | 11.46% |
| Reverse ejection | **186/192 — 96.88%** | 154/192 — 80.21% | **3.13%** | 8.85% |
| **Pooled** | **868/960 — 90.42%** [88.39%, 92.12%] | 784/960 — 81.67% [79.09%, 83.99%] | **26/960 — 2.71%** [1.85%, 3.94%] | 84/960 — 8.75% [7.12%, 10.71%] |

The confirmatory success gain is **+8.75 points** (Newcombe 95% CI **+4.40
to +13.03**), while violations fall by **6.04 points** (**3.18 to 8.85**).
The tradeoff is visible rather than averaged away: the hybrid gives up 6.25
nominal points and 9.90 forward-ejection points, but gains 13.54 on permanent
blockage, 29.69 on the reversible hard negative, and 16.67 on reverse
ejection. Independent permanent-specialist evaluation across three training
seeds reaches **1,477/1,536 — 96.16%**, with 3/1,536 violations.
Reverse-specialist training is less stable: after continuing each weak seed
from its own selected checkpoint for 15M additional transitions, three-seed
held-out performance is **1,321/1,536 — 86.00%**, with seed rates 94.34%,
83.59%, and 80.08% and 16/1,536 violations. The integrated table therefore
supports the frozen-controller result, not a claim that reverse-policy
training is seed-insensitive.

Renderer-only OOD evaluation was also frozen before execution (4 profiles ×
the same 960 episodes per method):

| Unseen renderer profile | Frozen hybrid success | V19 baseline success | Hybrid violations | V19 violations |
|---|---:|---:|---:|---:|
| Camera left 5 cm | **583/960 — 60.73%** | 202/960 — 21.04% | **10.21%** | 22.71% |
| Camera high 5 cm | **542/960 — 56.46%** | 23/960 — 2.40% | **9.90%** | 23.54% |
| Dim lighting | **622/960 — 64.79%** | 310/960 — 32.29% | **10.31%** | 21.25% |
| Warm lighting | **680/960 — 70.83%** | 435/960 — 45.31% | **8.02%** | 20.73% |
| **Pooled OOD** | **2,427/3,840 — 63.20%** [61.67%, 64.71%] | 970/3,840 — 25.26% [23.91%, 26.66%] | **9.61%** | 22.06% |

The OOD success difference is **+37.94 points** [35.01, 40.80], but this is
not a general visual-robustness claim: camera shifts still damage the nominal
and post-clearance RGB branches. This controller is also **not restricted
RGB**. Its router and specialists receive named object-state observations;
RGB, proprioception, instruction, and learned progress are retained for the
V19 branch. The reverse classifier was tested on a held-out mechanism, but the
reverse control specialist was trained for reverse ejection.

The machine-readable aggregation is produced by
[`scripts/aggregate_v4_publishable_results.py`](scripts/aggregate_v4_publishable_results.py)
from `results/v4_temporal_controller_v28_confirmatory`,
`results/v19_on_v4_v28_confirmatory`, and the two V4 OOD result trees. The
frozen classifier hash begins `28940882`; forward, permanent, and reverse
specialist hashes begin `7ca3ec24`, `b4a42a55`, and `c7bb71ad`. Environment,
controller, and contract checks live in
[`src/atr/envs/learned_recovery_v4.py`](src/atr/envs/learned_recovery_v4.py),
[`scripts/evaluate_v4_temporal_controller.py`](scripts/evaluate_v4_temporal_controller.py),
and [`tests/drafts/test_v4_temporal_controller.py`](tests/drafts/test_v4_temporal_controller.py).
The canonical confirmation and aggregation can be reproduced on Slurm with:

```bash
sbatch --export=ALL,ATR_V4_SEED_BASE=280000000,ATR_V4_CONTROLLER_OUTPUT=results/v4_temporal_controller_v28_confirmatory scripts/slurm_evaluate_v4_temporal_controller.sh
sbatch --array=0-44:3 --export=ALL,ATR_V4_SEED_BASE=280000000,ATR_V4_OUTPUT_DIR=results/v19_on_v4_v28_confirmatory scripts/slurm_evaluate_v19_on_v4.sh
python scripts/aggregate_v4_publishable_results.py
```

### Latest non-teleport visual-recovery result

V19 remains the established integrated restricted-input controller: across
three seeds and 768 held-out episodes per regime it reaches **96.35% strict**
and **91.41% nominal safe success**, with **97.06%/95.69%** on the two actual
physical-removal branches. The actor executes continuous joint control from
RGB, robot proprioception/TCP, the instruction, and learned visual progress;
object poses and evaluator domain labels are unavailable at deployment.

V41 preserves V19's controller and adds learned continuous/dense RGB
canonicalization behind a fixed magnitude gate. Its three-seed lineage passed
every checkpoint audit. On standard evaluation it reaches **687/768 — 89.45%
nominal** and **734/768 — 95.57% intervention safe success**, with an **83.20%**
minimum seed. On matched strict removal it exactly matches V19 at **740/768 —
96.35%**, with seed rates **98.05%, 96.88%, and 94.14%**.

The frozen untouched suite shows a real but incomplete robustness gain. V41's
mean across 14 new domain/condition cells is **44.47%**, up from V35's
**18.34%**. Synthetic geometry transfers best: intervention safe success is
**82.03%** for a 2.25-pixel shift, **78.13%** for a four-degree rotation,
**73.96%** for 1.08 scale, and **72.92%** for their combined transform. Learned
progress is causally useful across all three seeds: cyclic shifting its bits
reduces safe success by **11.07 points nominal** [4.43, 19.79] and **13.15
points under intervention** [0.39, 23.18].

This is still not a general-robustness success. The new combined camera shift
falls to **0.26% nominal / 10.81% intervention**, and opposite-side lighting
falls to **0% / 5.08%**. The frozen final gate passes **6/10** checks: standard
intervention, the standard seed floor, all three strict checks, and causal
utility. Standard nominal misses its 90% threshold by **0.55 point**, while
mean/minimum untouched robustness and the all-domain rule fail. The supported
finding is narrower and useful: **canonicalization materially improves unseen
geometric transfer without sacrificing strict recovery, but viewpoint and
directional-light generalization remain unresolved.** Training uses privileged
same-state supervision, so this is neither pure self-supervision nor
end-to-end pixel RL.

### Historical V2 state-control result and reward audit

The original integrated experiment learned recovery and continuous control in
one environment. A force-driven sweeper physically removes either the first or
second requested cube mid-episode; the same Panda PPO policy must infer what
remains possible, execute the feasible ordered suffix, and avoid moving a
protected object. All methods use the same continuous joint action space and
three 99,942,400-transition training runs. Each result below is 768 disjoint
held-out intervention episodes; **safe success** means task success with no
protected-object violation anywhere in the episode.

| Learned-control policy | Raw success | Violation | Safe success |
|---|---:|---:|---:|
| **Adaptive PPO** | 459/768 — 59.77% [56.26%, 63.18%] | 8.59% | **397/768 — 51.69%** [48.16%, 55.21%] |
| Privileged unavailable-state PPO | 500/768 — 65.10% [61.67%, 68.39%] | 20.83% | 354/768 — 46.09% [42.60%, 49.63%] |
| No-intervention-training PPO | 295/768 — 38.41% [35.04%, 41.90%] | 4.95% | 279/768 — 36.33% [33.00%, 39.79%] |

Adaptive minus no-intervention safe success is **+15.36 points** (paired
bootstrap 95% CI **[+10.68, +20.05]**). Adaptive also exceeds the privileged
policy on safe success by **+5.60 points** (**[+1.17, +10.03]**): the oracle's
higher raw completion is offset by more constraint failures. On first-goal
removal, adaptive achieves **33.24% safe success** versus **0%** for the
no-intervention policy; on second-goal removal they are matched at 67.80% and
68.05%. This is learned state control, not a vision-policy result, and all
three methods retain meaningful seed variance.

> **Important reward-audit boundary.** A later code audit found that
> `LearnedRecovery-v2` paid for the first completed goal on every remaining
> step. At `gamma=0.95`, waiting could yield more discounted return than
> completing the second goal and terminating. A fresh three-seed audit confirms
> the consequence: state PPO achieved **50.65%** on 768 forced-intervention
> episodes but only **1.95%** on 768 nominal two-goal episodes. The table above
> is retained as historical evidence about V2's intervention branches, not as
> the final visual-recovery result.

The isolated `LearnedRecovery-v3` benchmark keeps the same physics,
interventions, continuous controls, observations, ordering, and safety rules,
but uses transition-local progress and one-time completion rewards. Its
completed three-seed state recovery reference achieved **55.34% raw success**
(425/768), **55.21% safe success** (424/768), and **1.56% violations** on
held-out forced interventions. The same recovery-specialized policy scored
0/768 nominal successes, so it is not presented as a nominal-control result;
a separately trained nominal-only state control also failed to solve the task
robustly: **145/768 raw (18.88%)**, **131/768 safe (17.06%)**, and **16/768
violations (2.08%)**. Its seed-level safe rates are 0%, 4.30%, and 46.88%, so
the pooled score is not evidence of reliable nominal control. This negative
result is retained rather than selecting the one favorable seed.

The V3 restricted-RGB screening policy has now completed independent held-out
evaluation. Its actor uses a 64×64 camera image, robot proprioception/TCP, and
the factorized instruction—never object, goal, sweeper, or protected-object
poses. Across three training seeds it achieves **748/768 nominal raw success
(97.40%)**, **741/768 nominal safe success (96.48%)**, **708/768 forced-sweeper
raw success (92.19%)**, and **699/768 forced-sweeper safe success (91.02%)**.
Hierarchical seed/episode intervals are [95.96%, 98.70%] for nominal raw and
[87.63%, 94.53%] for forced safe success; forced-removal violations are 1.43%.
Only 125/768 sweeper-condition episodes actually made the selected goal
unavailable, because the fast controller often completed it before contact;
118/125 of that conditional subset were safe successes, but only five were
actual first-goal removals. Thus 92.19% is not presented as a balanced post-
removal recovery rate. In the separately frozen step-0 strict-removal test,
all 768/768 episodes contain recognized physical removal. The clean visual
policy achieves **404/768 raw success (52.60%)** and **402/768 safe success
(52.34%)**, with **5/768 violations (0.65%)**. The preregistered adaptive-visual
continuation has also completed: it retains **734/768 nominal raw (95.57%)**
and **723/768 nominal safe (94.14%)**, but falls to **255/768 strict raw
(33.20%)** and **249/768 strict safe (32.42%)**, with 13/768 strict violations
(1.69%). Its strict seed-safe rates are 5.47%, 50.78%, and 41.02%; therefore
the pooled result is not evidence of robust adaptation and does not confirm an
adaptive-recovery advantage over the clean policy.
The historical state recovery reference, trained for later 2 N interventions,
scores only **23/768 raw (2.99%)** and **22/768 safe (2.86%)** under the same
early-removal stress test, with 35.42% violations. The paired clean-visual minus
historical-state effect is +49.61 points [33.98, 61.59] raw and +49.48 points
[33.98, 61.07] safe. That apparent visual advantage is entirely a training-
distribution artifact: the matched strict-trained state PPO reaches **756/768
raw and safe success (98.44%)**, zero violations, 98.66% first-goal-removal
safe success, and 98.22% second-goal-removal safe success. On identical seeds,
clean visual trails it by 46.09 safe-success points [−58.33, −36.20]. The
state result is the current matched-distribution upper baseline. The integrated
V13 restricted-RGB extension closes most of the gap while retaining nominal
control: **697/768 nominal safe (90.76%)** and **689/768 strict safe (89.71%)**,
with 2.47%/1.17% violations. It is close but ineligible under the frozen gate:
strict safe misses 90% and first-goal-removal safe is 83.69%, below 85%. The
second-removal branch is 95.43%. No threshold was relaxed and five-seed
confirmation received zero allocation.

The frozen visual hypothesis report is deliberately less flattering than the
screening result: primary direct RGB V1, asymmetric-training V2, temporal-loss
V3, and adaptive-recovery V4 are all rejected. V5 passes only its originally
locked comparison against a historical 2.86%-safe state reference; it does not
match the later strict-trained state PPO at 98.44% safe. The strong DAgger RGB
fallback establishes restricted-input competence but cannot retroactively
rewrite V1--V4 or support a competitive V5 claim.
The upper baseline is not an integrated solution: on a separate 768-episode
nominal evaluation it records **0 successes and 73.57% violations**. Thus its
98.44% strict score is a distribution-specialized ceiling, not evidence that
one state policy solves both regimes. A preregistered proposal to use it as a
DAgger teacher failed its nominal-competence gate and consumed no V14 training
allocation.
A new privileged-input state control completed training on the exact V13 distribution:
80% strict / 20% nominal training with balanced 50/50 checkpoint selection and
separate final endpoints. This is the fair integrated state upper baseline;
neither the strict specialist nor the historical state result can substitute
for it. All three seeds reached 99,942,400 floor-aligned transitions and passed
the finite checkpoint audit. Frozen held-out evaluation confirms a one-regime
solution: **748/768 strict safe successes (97.40%)** with 15 violations
(1.95%), but **0/768 nominal raw or safe successes**. The preregistered teacher
gate therefore failed only its nominal-safe threshold and allocated no V15/V16
RGB training. A disclosed failure-only reverse curriculum initialized from the
strict specialist also failed by catastrophic nominal forgetting: 92.58%
strict safe, 91.18%/93.91% branch safe, and 0.91% violations, but 0/768 nominal
successes. Because both independent integrated-state teacher routes failed,
the fail-closed router released the V19/V20 dual-specialist RGB fallback pair;
no integrated-state claim is made.
V19 then completed three exact 100M-step seeds and passed every frozen held-out
gate: **96.35% strict safe success**, **91.41% nominal safe success**,
**97.06%/95.69%** safe success on the two physical-removal branches, and
**1.30%/3.65%** strict/nominal violations (768 episodes per regime). It is the
first eligible integrated restricted-input visual policy, with continuous
control and no teleportation. Training uses privileged dual teachers, progress
labels, and an asymmetric critic, so this is not a pure pixel-RL or pure
self-supervised claim;
the strict state policy remains a stronger upper bound at 98.44% safe success
and zero violations. Full-strength VICReg V20 improves matched-pixel pose and
goal-resolution R² by +0.0106 and +0.0146 over V19, but fails control selection
at 85.42% strict safe and 74.06% first-removal safe success. This negative
ablation shows that improved linear decodability does not guarantee improved
recovery. In the matched seven-method table, V19 is the only visual or state
method above 90% at its worst strict/nominal/removal endpoint (91.41%); all
three state curricula score 0% nominal safe despite strong strict recovery.
This is an in-benchmark simulation result, not cross-benchmark or real-robot
superiority. V19 uses 99.999M PPO plus 1.92M DAgger interactions per seed;
upstream initializer and teacher training is disclosed separately rather than
hidden in that new-stage count. The preregistered lower-variance V21 extension
is rejected: it reaches 92.19% nominal safe success but only 87.63% strict and
78.34% first-removal safe success. The clean
policy's identical-pixel linear pose probe is negative—learned-minus-random R²
is −0.177 [−0.334, −0.037]. The adaptive V7 encoder is positive on its separate
probe, +0.387 [0.312, 0.488] (learned R² 0.725 versus random 0.339), with all
three seed differences positive. V7 jointly uses a temporal loss, privileged
pose auxiliary supervision, and supervised progress labels, so this is only
linear decodability evidence; it cannot be attributed to self-supervision or
used as a causal control-performance claim. See
[`docs/14-results-and-claim-boundaries.md`](docs/14-results-and-claim-boundaries.md),
[`docs/16-visual-recovery-hypotheses.md`](docs/16-visual-recovery-hypotheses.md),
[`docs/17-visual-training-ledger.md`](docs/17-visual-training-ledger.md), and
[`docs/18-evidence-blueprint.md`](docs/18-evidence-blueprint.md).

The integration gap is now tested directly in one non-teleport Fetch episode.
One parsed instruction asks for the can and cracker box while protecting the
master-chef can; the cracker is irreversibly removed during physical can
execution. A fixed RGB change detector feeds a reward-trained Q policy, the
intent/navigation guard screens accepted actions, and the contact-verified
Fetch controller executes them. Thirty paired seeds per policy give:

| Integrated Fetch policy | Achievable can completion | Wasted physical steps | Violations |
|---|---:|---:|---:|
| Static | 63.3% [46.7%, 80.0%] | 461.6 [370.4, 552.9] | 0/30 |
| Privileged oracle | 60.0% [43.3%, 76.7%] | 191.6 [111.8, 271.4] | 0/30 |
| **Visual learned + guard** | **76.7%** [60.0%, 90.0%] | **111.8** [47.9, 191.6] | **0/30** |

Visual feasibility is 30/30 for every policy and the audit records zero
teleport calls across all 90 episodes. Learned versus static wasted-step
difference is **−349.9** (paired-bootstrap 95% CI **[−470.8, −226.8]**).
More directly, static continuation spends 285.9 steps on the destroyed goal
[274.6, 297.3], while oracle and learned spend exactly 0.
The completion difference is not statistically resolved, so the supported
claim is recovery efficiency at matched safety—not higher completion.

**How this held together:** every comparison is paired-seed and bootstrapped
(docs/10's predeclared protocol, used consistently); every required baseline
(domain-randomized, symbolic replanner, imitation learning, frame-difference
detector) is actually built and compared, not asserted as future work; the
full test suite (real simulator episodes, no mocks on load-bearing paths)
runs before anything is called done, and has caught real regressions targeted
tests missed; and negative results are kept, not edited out — a CLIP
calibration that didn't transfer, a hypothesis that only holds conditionally,
a config value that never did what its own comment claimed, all reported as
found, with the fix or the disclosed gap next to it.

**Five capabilities beyond the five hypotheses, each separately scoped:**

- **Real pick-and-place on Fetch** (D-124, D-130): navigate, reach with real closed-
  loop inverse kinematics, grasp with a real contact-force check
  (`agent.is_grasping()`), carry across the apartment while still gripping,
  place, and release — verified with the project's own `goal_achieved()`
  check, not a custom one. In the current sequential 10-episode evaluation,
  the can is physically grasped and placed in 10/10 episodes, while the bowl
  grasp fails in 10/10: 1.0/2.0 mean goals and 0/10 complete tasks. This
  negative result is kept explicit. The controller remains separate from the
  abstract policy benchmark.
- **A cluster-ready scaled benchmark contract** (D-125): versioned manifests,
  content-addressed cases, resumable sharded execution, strict pairing/
  completeness validation, and stratified bootstrap CIs — built because the
  prior harness was correct for small in-process comparisons but couldn't
  safely run at scale. Full v1 expands to 3,200 cases / 12,800 paired policy
  episodes across all four environment families, all three ReplicaCAD
  layouts, and 100 seeds. The frozen v1 run is complete: 3,200 paired cases
  and 12,800 policy episodes. Oracle feasibility and static execution both
  achieve 1.68625 goals/case, while static execution wastes 14.24 additional
  steps per paired case (95% paired-bootstrap CI 12.708–15.842). The original
  v1 safety column is invalid because of an evaluator asymmetry and is not
  used; a corrected 2,000-episode effect-aware safety benchmark is reported
  separately in the result index.
- **Learned non-teleport manipulation** (D-132): three-seed state-PPO on
  standard ManiSkill tasks, 50M requested transitions per seed and 256
  independent held-out episodes per seed. Pooled held-out success is 98.31%
  for PickCube (755/768), 69.01% for randomized PickSingleYCB (530/768), and
  99.87% for Unitree G1 apple-in-bowl (767/768). This establishes continuous
  control on those tasks; it does not imply transfer to the Fetch apartment or
  turn the abstract adaptation benchmark into physical manipulation.
- **Integrated non-teleport recovery** (D-134): parsed language, fixed-camera
  RGB change detection, a Q policy trained through the physical Fetch
  executor, intent/navigation guarding, an irreversible object removal, and
  contact-verified can manipulation now run in one episode. The learned policy
  significantly reduces wasted execution relative to static continuation.
  The RGB detector is calibrated to one object/view and the low-level Fetch
  skill is scripted, not a learned motor policy.
- **Integrated learned-control recovery** (D-135): one factorized-instruction
  PPO policy controls Panda joints directly while a force-driven intervention
  removes either ordered goal. Across 768 held-out intervention episodes,
  adaptive training improves safe success over no-intervention training by
  15.36 points with a paired interval excluding zero, including a 33.24-point
  gain on the first-goal-removed branch. No pose assignment occurs after reset;
  the result uses state observations and remains simulation-only.

**What's still genuinely open:** the historical V2 learned-control recovery
policy uses privileged simulator state and has a success-delaying reward
defect. The corrected V3 RGB policy now has a strong three-seed held-out
screen, but its matched V7 adaptive continuation is worse under actual removal
and five-seed confirmation remains gated on V13. The clean frozen encoder does
not beat a random encoder on the linear pose probe; V7 does, but its objective
mix prevents attributing that gain specifically to self-supervision.
Collision geometry in the high-level safety screen is still spheres and points,
not full robot/object meshes; CLIP
calibration is scene-specific and does not transfer automatically; the Fetch
controller has not solved the bowl grasp or a task where both requested Fetch
objects remain achievable; everything is simulation-only, with no real-robot
result; and the integrated pipeline uses a calibrated frame-difference signal
plus a scripted low-level Fetch skill rather than a self-supervised visual
encoder plus learned continuous motor policy. These are disclosed scope, not
benchmark claims.

That gap also has a pixel-based NE-Dreamer V2 pilot. The decoder-free world model optimized its
self-supervised next-embedding objective (83.2% mean loss reduction), but the
continuous recovery policy achieved **0/768** final held-out successes after
250k environment steps per seed. Actions were bounded, all metrics were
finite, and the exact two-sided 95% upper bound is 0.48%, so this is a real
negative control-learning result rather than a crashed run. Because it used V2
and a smaller 250k-step budget, it is context rather than a matched V3 baseline. See
[`docs/15-vision-ne-dreamer-pilot.md`](docs/15-vision-ne-dreamer-pilot.md) for
the protocol, upstream action/replay bug found and fixed, and next experiment.

### What this project does not claim

The Fetch recordings above are real, not scripted: real navigation, real
collision-aware replanning, real inverse kinematics, and a real contact-force-verified grasp
(`agent.is_grasping()`, ManiSkill3's own detector — checked at every stage,
not assumed). The pick-and-place demo is built in a separate, additive module
(`src/atr/envs/tidy_up_replicacad_manipulation.py`, D-124), deliberately kept
apart from the `attempt_goal()` navigate-then-teleport contract every H1-H5
result and every navigation-safety decision (D-091–D-123) is built on across
300+ regression tests — that contract stays exactly as it was; changing it project-wide
for a demo's visual benefit would risk the whole evidence base for no
research reason. So concretely:

The hero montage replays the frozen V19 seed-4796 checkpoint selected at
96,657,408 steps. That seed achieves 98.44% strict and 94.14% nominal safe
success in the held-out evaluation. Both recovery panels use the locked 6 N,
24-step intervention and are labeled as removed goals only after capture
metadata verifies recognized physical unavailability, safe success, the exact
strict-config and selector hashes, and zero teleport calls. The first-goal
panel is the first qualifying episode in the declared seed range (seed
92,000,001); the second-goal and nominal panels use seed 92,000,000. The other
learned-control montage panels replay
frozen PPO checkpoints on standard ManiSkill tasks. Those use continuous
actions and no ATR teleport executor, but do not include ATR's language goals,
interventions, or Fetch apartment.

- **What's real:** Fetch, the potted meat can, one scene —
  navigate, reach, grasp, lift, carry across the apartment while still
  gripping, place, release, and a real physics-settled final position,
  verified with the project's own `goal_achieved()` check, not a custom one,
  in 10/10 sequential episodes.
- **What isn't (yet):** the scripted bowl grasp failed in all 10 episodes, so
  the complete two-achievable-object physical task remains unsolved. D-134
  wires this controller into a separate integrated recovery policy, but it
  does not retroactively change the H1–H5 benchmarks and doesn't cover
  the humanoid embodiment — a real analytic-Jacobian IK check there
  (`src/atr/control/ik_solver.py`) found, and kept as a disclosed regression
  test rather than hidden, that neither goal object is within true contact
  range from G1's calibrated standing position. That's a measured kinematic
  limit of that setup, not a missing feature, and it's why this capability
  was built for Fetch specifically rather than assumed to transfer.

## Planned system

- A language parser that represents goals, priorities, and hard constraints
- A self-supervised visual encoder that learns object and scene representations
- A world-change and goal-feasibility estimator
- A language-conditioned RL policy that replans or changes strategy
- An intent guard that rejects actions that contradict the instruction
- A benchmark with controlled irreversible changes and held-out changes
- Evaluation against static-policy, replanning, and representation baselines

See [`docs/03-system-architecture.md`](docs/03-system-architecture.md) for
the module-boundary and ownership diagram behind the list above.

## Scope and status

The first version is simulation-only and targets a **simulated humanoid** carrying
out multi-goal, visually observable, object-centric tasks. The feasibility and
intent modules are embodiment-agnostic, while a humanoid control layer provides
navigation, reaching, grasping, and safe whole-body skills. A simpler embodiment
may be used for early research debugging, but humanoid evaluation is a required
project milestone.

The core task schema — `Goal`/`Constraint`/`GoalGraph`, oracle feasibility,
the intent guard — is accepted and promoted to [`src/atr/`](src/atr/) (D-037).
**Read the review's status banner before treating that as settled**: it was
self-resolved by the project owner, not independently reviewed by the
teammate it was written for — see
[`ai-notes/review-request-task-schema.md`](ai-notes/review-request-task-schema.md).
The promotion sweep is effectively complete: language parsing, zero-shot
CLIP feasibility, every required baseline (static/oracle-feasibility/naive-
substitution/domain-randomized/symbolic-replanner/imitation-learning),
Q-learning, every embodiment/scene environment (a Panda arm, a Unitree G1
humanoid, a real ReplicaCAD apartment with a mobile Fetch robot including a
full collision-aware navigation-safety stack, and G1 placed in that same
apartment across three independently verified scene layouts), the end-to-end
pipeline, the evaluation harness, a queryable dataset-split registry
(instruction-, intervention-, and scene-layout-level), a log interface,
experiment tracking, real pick-and-place on Fetch (D-124, see the demo GIF
above), and the completed cluster-scale benchmark contract/run (D-125--D-126)
all live in `src/atr/` as tested architecture. Checkpointed non-teleport PPO
and held-out evaluation for three standard ManiSkill manipulation tasks are
complete: 2,304 held-out episodes across nine trained policies, with pooled
success reported above. Only `dinov2_probe.py` remains spike-stage in
`spikes/task_schema_draft/` — DINOv2 wired into a real live decision loop,
a genuine robustness gap found and closed (D-054/D-055), still not
promotion-ready. See the result index for the latest validation status rather
than relying on a historical test count.
See [STATUS.md](STATUS.md) for current work, [`ai-notes/decisions.md`](ai-notes/decisions.md)
for the full decision-by-decision record, and [docs/](docs/) for the study design.

## Roadmap

1. Select a humanoid-capable environment and define the goal/constraint task schema.
2. Build deterministic irreversible-world-change interventions.
3. Train and compare self-supervised visual representations.
4. Learn goal-feasibility prediction and calibrated uncertainty.
5. Train the adaptive vision-language RL policy and intent guard.
6. Evaluate compositional and out-of-distribution generalization.

See the [full roadmap](docs/11-roadmap-and-milestones.md).

## Scope areas

The benchmark, schemas, integration contracts, and final evaluation are
shared foundations. Two work areas sit on top of that foundation:

- **Representation and feasibility:** visual/language model selection,
  self-supervised representations, goal graphs, change inference, per-goal
  feasibility, calibration, and abstention.
- **Policy and humanoid execution:** simulator/asset integration,
  low-level skills, static and adaptive RL policies, intent guard, and policy baselines.

Integration is continuous: policy work develops against oracle feasibility,
representation work develops against recorded trajectories, and both
replace the oracle with learned beliefs through a versioned shared interface.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
