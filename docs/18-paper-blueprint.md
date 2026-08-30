---
title: Paper Blueprint and Evidence Contract
status: active-experiment
last_updated: 2026-08-28
---

# Recover What Remains: Visual Manipulation after Irreversible Goal Loss

This document is the paper-writing contract for the non-teleport V3 track. It
is deliberately incomplete while the preregistered adaptive and confirmation
cohorts run. A number enters the abstract, contribution list, or conclusion
only if its named aggregate below exists and passes its fail-closed verifier.

## One-sentence thesis

A single continuous, language-conditioned visual policy can detect that an
ordered manipulation goal has become physically unavailable, abandon only that
goal, complete the feasible suffix, and respect an explicit protected-object
constraint—without object-pose input or runtime teleportation.

## Candidate contributions

1. **Controlled recovery benchmark.** A GPU-batched two-stage manipulation
   environment in which a contact-driven intervention makes one requested goal
   physically unavailable and success requires ordered feasible-suffix
   completion under a protected-object constraint.
2. **Restricted visual recovery policy.** A 64×64 RGB, robot-proprioception,
   TCP, and factorized-instruction actor; object, goal, sweeper, protected-
   object, and oracle progress poses are excluded at deployment. Privileged
   critic state, teacher actions, and labels are training-only and disclosed.
3. **Causal evaluation protocol.** Matched reset seeds, no teleport calls,
   policy-independent step-0 contact intervention, required recognized physical
   unavailability in every strict episode, branch-balanced reporting, safety-
   qualified success, and hierarchical training-seed/episode uncertainty.
4. **Mechanism and baseline study.** Direct RGB PPO, asymmetric training,
   temporal self-supervision, DAgger, learned progress, state PPO, clean-to-
   adaptive continuation, and post-audit matched-distribution extensions.

Contribution 3 may be claimed now. Contributions 1–2 require the final strict
adaptive aggregate. Contribution 4 must report negative and unstable variants,
including the linear-probe failure and V12 catastrophic nominal forgetting:
97.40% strict safe success but 0/768 nominal successes under frozen evaluation.

The no-teleport contract has both static and runtime evidence. AST inspection
requires every environment `set_pose` call to occur only in episode reset; the
Jarvis integration test then resets `LearnedRecovery-v3`, replaces ManiSkill's
`Actor.set_pose` with a function that raises, and executes five intervention
steps successfully. Thus task execution and the physical intervention do not
assign poses after reset. The combined learned-recovery, visual-policy, and
statistical/checkpoint/teacher/selection-gate/method-accounting contract suite passes 100/100 tests
on Jarvis after the final V3 runtime-test, checkpoint gate, V14 allocation gate,
integrated-policy selection, dual-regime comparison, and strict five-seed
allocation changes.

## Paper structure

1. **Introduction:** persistent changes can invalidate intent rather than merely
   perturb a trajectory; recovery may require abandoning an impossible stage.
2. **Related work:** skill-level replanning, learned recovery, visual long-
   horizon manipulation, model-based visual RL, self-supervised control, and
   constrained RL. Use `02-background-and-related-work.md` and
   `references.md`; do not compare percentages across different task suites.
3. **Problem:** ordered goal sequence, latent feasibility, irreversible
   intervention, feasible suffix, hard protected-object constraint, and partial
   observability.
4. **Method:** visual encoder, factorized instruction, learned progress head,
   asymmetric critic, DAgger initialization, temporal latent prediction, PPO
   continuation, and checkpoint selection.
5. **Benchmark:** randomized cube starts, fixed camera, physical sweeper,
   continuous Panda control, V3 event reward, nominal/sweeper/strict protocols.
6. **Experiments:** hypotheses V1–V5, ablations, strict branch analysis,
   state-policy controls, five-seed confirmation, representation diagnostics,
   and compute accounting.
7. **Limitations:** simulation-only evidence, factorized rather than open-ended
   language, one camera and embodiment, custom benchmark, training-only
   privilege, and no direct cross-benchmark numerical comparison to published
   systems.

## Authoritative result artifacts

| Claim block | Required artifact | Status |
|---|---|---|
| Nominal restricted-RGB competence | `results/visual_recovery_ppo/visual_recovery_progress_dagger_v6_event_reward/aggregate.json` | Complete |
| Clean strict physical removal | `results/strict_removal_comparison/strict_removal_screening_comparison_v1/aggregate.json` | Complete |
| Strict state-training effect | `results/strict_removal_comparison/strict_removal_state_training_comparison_v1/aggregate.json` | Complete |
| Preregistered V1–V5 | `results/final_visual_comparison/hypotheses.json` | Complete: V1–V4 rejected; V5 confirmed only against the frozen historical state reference and explicitly not competitive with the post-audit 98.44%-safe strict state baseline |
| Full strict comparison | `results/strict_removal_comparison/strict_removal_extension_comparison_v1/aggregate.json` | Dependency-scheduled |
| Strict paper table/figure | `results/paper/strict_removal_extension.{md,csv,png,pdf,metadata.json}` | Dependency-scheduled |
| Matched strict/nominal paper comparison | `results/paper/integrated_regime_comparison_v2.{json,md,csv,png,pdf}` | Complete as `1140913`/`1140914`; seven methods, identical three-seed/768-episode protocols, exact source hashes; V19 only >90% worst endpoint |
| Matched interaction accounting | `results/paper/integrated_sample_efficiency_v1.{json,csv,md}` | Complete as corrected job `1140927`; PPO/DAgger/new-stage costs per seed, upstream lineage disclosed, no invented normalized score |
| Five-seed integrated confirmation | `results/paper/integrated_regime_five_seed_comparison_v2.{json,md,csv,png,pdf}` | Strict/nominal V13-gated as chain `1139772`--`1139786` |
| Integrated-teacher RGB extension | state-only gate aggregate `results/strict_removal_comparison/strict_removal_integrated_state_teacher_gate_v1/aggregate.json`; `results/strict_removal_comparison/strict_removal_integrated_teacher_extension_v5/aggregate.json`; `results/gates/integrated_visual_selection_v3.json` | State-only aggregate `1139845`, teacher gate `1139804`, extension `1139805`--`1139815`; privileged training must be disclosed |
| Failure-only state continuation | `results/gates/state_fallback_release_v1.json`; `results/gates/integrated_from_strict_state_teacher_v2.json` plus its strict/nominal source aggregates | Router `1139959` requires an actual failed artifact from scratch integrated teacher gate `1139804` before `1139850`--`1139857` can allocate; strict-initialized 20% strict / 80% nominal reverse curriculum, post-registration fallback, never mixed with a passing or operationally missing scratch result |
| Failure-only reverse-teacher RGB pair | `results/strict_removal_comparison/strict_removal_reverse_teacher_extension_v7/aggregate.json`; `results/gates/integrated_visual_selection_v5.json`; matched pose/task-semantic representation reports | `1139860`--`1139883` run only after fallback teacher gate `1139857` passes; V17/V18 privileged-training extension, zero allocation on the primary passing path |
| Failure-only dual-specialist RGB | `results/gates/dual_specialist_teacher_v1.json`; `results/gates/dual_specialist_release_v1.json`; `results/strict_removal_comparison/strict_removal_dual_specialist_extension_v8/aggregate.json`; `results/gates/integrated_visual_selection_v6.json`; matched pose/task-semantic representation reports | Teacher gate `1139917` passed all seven frozen specialist checks. Router `1139957` releases `1139901`--`1139915` only when both state-gate artifacts exist and report failure and V13 passes its checkpoint audit. V19 routes a nominal RGB and strict state teacher using a training-only physical-resolution label, with RGB-only deployment |
| Failure-only dual-specialist VICReg ablation | `results/gates/dual_specialist_release_v1.json`; `results/strict_removal_comparison/strict_removal_dual_specialist_vicreg_extension_v9/aggregate.json`; `results/gates/integrated_visual_selection_v7.json`; matched V20–V19 pose/task-semantic reports | `1139931`--`1139945` share V19's completed teacher gate and fail-closed router. V20 is an exact V19 ablation except variance/covariance penalties; after both state/RGB fallback edges were made artifact-driven, the 24-hour signal target was corrected, and the strict-eval alias was regression-tested, the expanded 100-test suite passed as `1140069` |
| Low-variance VICReg stabilization | `results/gates/vicreg_low_variance_smoke_gate_v1.json`; `results/strict_removal_comparison/strict_removal_dual_specialist_low_variance_extension_v10/aggregate.json`; `results/gates/integrated_visual_selection_v8.json` | The one-seed, matched-budget allocation gate passed all three frozen checks at 19,996,672 scheduled steps; this is training-stream allocation evidence only. The independently frozen three-seed 100M V21 array and held-out chain are `1140381`--`1140395` and remain incomplete |
| Continuation-stage temporal SSL control | V26 config; strict comparison V15; `temporal_ssl_continuation_ablation_v1.json`; `integrated_regime_temporal_ablation_v3.json` | Frozen before V26 metrics. Jobs `1140929`--`1140947` cover exact training/audit, held-out regimes, matched report, and paired Boolean verdict. Both arms inherit upstream temporal/privileged training; only continuation-stage coefficient 0.01→0.0 is isolated |
| Anti-collapse SSL ablation | `results/strict_removal_comparison/strict_removal_vicreg_extension_v6/aggregate.json`; `results/gates/integrated_visual_selection_v4.json`; `results/paper/visual_representation_vicreg_ablation_v5/representation_comparison.{json,md}` | Smoke- and teacher-gated V15/V16 matched ablation `1139819`--`1139830`; control and encoder claims remain separate |
| Task-semantic representation ablation | `results/paper/visual_task_representation_vicreg_ablation_v1/task_representation_comparison.{json,md}` | V15/V16 byte-identical-pixel goal-resolution probe `1139833`--`1139837`; supervised progress labels and non-causal claim boundary must be explicit |
| Method information and interaction accounting | `results/paper/method_information_contract_v1.{json,csv,md}` | Complete as V20-inclusive replacement job `1139946`; 20 methods, exact new-stage PPO/DAgger counts, training privilege, and initializer/teacher lineage; protocol artifact only, not performance evidence |
| Selected V19 qualitative montage | `media/demos/learned-recovery-montage.gif`; candidate `learned-recovery-montage-v19.gif`; three capture JSON files in `results/visual_recovery_ppo/videos_v19/` | Selection-gated capture `1140898`, candidate `1140899`, promoted hero `1140903`; all branches safe, intervention branches verify actual removal, zero teleport calls, sampled frames inspected |
| Representation diagnostic | `results/paper/visual_representation_strict_stability_matched_v3/representation_comparison.{json,md}` | Complete three-seed, byte-identical-pixel V13-versus-V6 comparison. V13 has +0.0330 mean R² relative to V6, paired seed-bootstrap interval [0.0141, 0.0619], but both learned encoders have negative mean R² and neither reliably beats its random control; relative decodability only |
| Causal-head and visual-OOD suite | `configs/v19_incumbent_causal_ood_v1.json`; `configs/selected_visual_causal_ood_v1.json`; `results/gates/visual_ood_rendering_preflight_v1.json`; `results/paper/v19_incumbent_causal_ood_v1/aggregate.{json,md}` | V19 incumbent complete as `1140989`/`1140990`: cyclic progress shift causes a +14.32-point baseline-minus-variant intervention safe-success drop [0.65, 29.69], confirming causal utility. Frozen visual-OOD robustness is rejected; all seven pixel/camera/lighting variants fail the joint 75%-safe/15-point-drop rule. Corrected final-selector jobs `1140991`/`1140992` remain held on V21. Simulation-only and no real-robot claim |
| Generic robust-distillation development | V27 smoke/full configs; `results/paper/v27_smoke_development_ood_v1/aggregate.{json,md}`; `results/gates/v27_robust_distill_smoke_gate_v1.json` | One-seed smoke retained 85.94%/87.89% nominal/intervention safe success but improved mean matched-seed OOD by only 4.69 points, left worst OOD at 0%, and regressed camera-left intervention by 20.70 points. Corrected matched-seed gate rejects it; three-seed allocation suppressed. Development-only negative result, not a paper robustness claim |
| DrAC-style stability ablation | `configs/drac_stability_smoke_gate_v1.json`; frozen V22 source-hash manifest; preserved runtime/smoke logs | Separate policy-consistency implementation preserves PPO likelihood ratios on original observations and applies stopped-target Gaussian KL only to shifted images. Runtime `1140573` collapsed; full-DAgger smoke `1140574` reached KL 1.86e20 and two zero-success evaluations before being stopped at 1.64M. Router `1140598` marked it ineligible, so gate/full/held-out chain `1140575`--`1140582` remains unallocated. V22 is a disclosed negative result |

## Main quantitative tables

### Table 1: Information and compute contract

For every method report actor inputs, critic inputs, teacher/labels, temporal
SSL, training seeds, PPO transitions, DAgger transitions, initializer cost, and
total protocol interactions. Never describe asymmetric/teacher-initialized
policies as pure pixel RL.

| Method | Deployed actor | Training-only privilege | Temporal SSL | Online DAgger/BC per seed | PPO transitions per seed | Total interactions per seed |
|---|---|---|---:|---:|---:|---:|
| Direct RGB PPO | RGB + qpos/qvel + instruction | None | No | 0 | 39,993,344 | 39,993,344 |
| Asymmetric RGB PPO | RGB + qpos/qvel + instruction | State critic | No | 0 | 39,993,344 | 39,993,344 |
| Asymmetric temporal RGB PPO | RGB + qpos/qvel + instruction | State critic | Yes | 0 | 39,993,344 | 39,993,344 |
| Clean learned-progress DAgger | RGB + qpos/qvel + TCP + instruction + predicted progress | Fixed state teacher, state critic, pose/progress labels | Yes | 1,920,000 | 39,993,344 | 41,913,344 |
| Preregistered adaptive visual | Same restricted visual actor | Clean initializer plus state critic and pose/progress labels | Yes | 1,920,000 initializer | 39,993,344 initializer + 99,999,744 continuation | 141,913,088 |
| V13 stable strict visual | Same restricted visual actor | Clean initializer, state critic, progress labels; pose head retained at zero loss | Yes | 1,920,000 initializer | 39,993,344 initializer + 99,999,744 continuation | 141,913,088 |
| V14 strict-teacher DAgger visual | Same restricted visual actor | Clean initializer, matched strict state teacher, state critic, progress labels | Yes | 1,920,000 initializer + 1,920,000 strict DAgger | 39,993,344 initializer + 99,999,744 continuation | 143,833,088 |
| V19 dual-specialist visual | Same restricted visual actor | Nominal RGB and strict state teachers, physical-resolution routing label, state critic, progress labels | Yes | 1,920,000 initializer + 1,920,000 routed DAgger | 39,993,344 initializer + 99,999,744 continuation | 143,833,088 |
| V20/V21 dual-specialist VICReg | Same restricted visual actor | V19 privilege plus variance/covariance training losses; V21 changes only variance coefficient | Yes | 1,920,000 initializer + 1,920,000 routed DAgger | 39,993,344 initializer + 99,999,744 continuation | 143,833,088 |
| V22 DrAC-style dual-specialist | Same restricted visual actor | V19 privilege plus training-only shifted-image policy-consistency target | Yes | 1,920,000 initializer + 1,920,000 routed DAgger | 39,993,344 initializer + 99,999,744 continuation | 143,833,088 |
| Strict-trained state PPO | Simulator state + instruction | State is deployed input | No | 0 | 99,942,400 | 99,942,400 |
| Integrated-mixture state PPO | Simulator state + instruction | State is deployed input | No | 0 | 99,942,400 | 99,942,400 |

All totals count environment transitions, not gradient updates. V14 remains
ineligible unless its teacher gate passes; listing its frozen budget does not
imply that the allocation ran. Initializer interactions are counted again for
each downstream method because each reported policy depends on that training
cost, even though the checkpoint is reused computationally.

### Table 2: Nominal and sweeper-condition control

Report raw success, safe success, violations, hierarchical 95% interval, and
per-seed rates. Call the intervention column **forced sweeper condition**, not
recovery, because only 125/768 clean episodes produced actual unavailability.

### Table 3: Strict physical-removal recovery

Use only `strict_removal_extension_comparison_v1/aggregate.json`. Report pooled
raw/safe success, first-goal-removed safe success, second-goal-removed safe
success, violations, hierarchical intervals, and matched paired effects. Every
included episode must have `goals_unavailable >= 0.5`; filtering is forbidden.

The primary integrated-policy comparison is generated separately from
`integrated_regime_comparison_v1.json`. It reports strict and nominal safe
success, both strict removed-goal branches, violations, hierarchical intervals,
and the minimum safe-success endpoint for every visual and state cohort. The
builder fails closed unless every source contains exactly 768 episodes from
three training seeds under the V3 event-reward semantics.

### Table 4: Hypotheses and ablations

Report primary verdicts separately from post-registration extensions. A
favorable DAgger or matched-distribution extension cannot turn a rejected
primary direct-RGB hypothesis into a confirmation.

## Current evidence that may be quoted

- Clean restricted actor nominal: 748/768 raw (97.40%), 741/768 safe
  (96.48%).
- Clean strict physical removal: 404/768 raw (52.60%), 402/768 safe
  (52.34%), five violations; 768/768 actual removals.
- Clean strict branch asymmetry: 109/374 safe when the first goal is removed
  (29.14%) versus 293/394 safe when the second is removed (74.37%).
- Matched strict-trained state PPO: 756/768 raw and safe (98.44%), zero
  violations, with 98.66%/98.22% safe success on first/second removal. Clean
  visual trails it by 46.09 safe-success points [−58.33, −36.20].
- The same strict-trained state checkpoints score 0/768 nominal raw/safe with
  565/768 violations (73.57%). They are a strict-specialist ceiling, not an
  integrated-policy result.
- DAgger factorial nominal: 79.56% without temporal SSL versus 97.27% with
  temporal SSL. The +17.71-point pooled extension passes its declared pooled
  threshold, but its paired hierarchical interval includes zero.
- Primary direct RGB factorial nominal: 0.26% safe for symmetric PPO and 0%
  for both asymmetric variants. Primary V1, V2, and V3 are rejected; the later
  DAgger competence extension cannot rewrite them.
- V7 adaptive continuation: 94.14% nominal safe but only 32.42% strict safe;
  clean-minus-V7 strict safe is +19.92 points [−0.13, 48.83]. It does not
  confirm an adaptive-recovery advantage.
- V13 integrated RGB: 90.76% nominal safe and 89.71% strict safe, with
  83.69%/95.43% removal-branch safe success and violations below 2.5%. It
  narrowly fails the frozen 90% strict and 85% first-branch thresholds and is
  not eligible; no five-seed allocation occurred.
- V13 task-semantic probe: learned-minus-random balanced accuracy +0.044
  [0.040, 0.050], ROC AUC +0.019 [0.007, 0.029], and R² +0.228 [0.131,
  0.344]. Pose R² gain is neutral. Supervised progress labels limit this to
  task-semantic decodability, not pure SSL or causal control evidence.
- Clean identical-pixel linear pose probe: learned-minus-random R² −0.177 with
  seed-bootstrap interval [−0.334, −0.037]. The separate V7 probe is +0.387
  [0.312, 0.488], but V7 combines temporal SSL, privileged pose auxiliary
  targets, and supervised progress labels; report decodability, not temporal-
  SSL attribution or causal control benefit.

The historical state policy's 2.99% strict score is a distribution-shift
diagnostic, not the final privileged baseline. The strict-specialist
replacement reaches 98.44% safe success with zero violations and must be the
primary privileged baseline for the strict-only table, while its 0% nominal
success disqualifies it from the integrated-policy table. The matched
integrated-mixture state policy was the primary privileged candidate for that
dual-regime table, but it completed with 97.40% strict safe success and 0%
nominal success. The reverse-curriculum state fallback likewise retained
92.58% strict safe success, 91.18%/93.91% branch-safe success, and 0.91%
violations, but scored 0% nominal. Both state routes therefore fail the frozen
integrated gate. This distinction is reported prominently rather than using a
mismatched or regime-specialized state policy to imply visual superiority.
The stabilized V13 extension trains one integrated policy on an 80% locked
strict-removal / 20% nominal mixture and is evaluated independently on both
conditions. Checkpoint selection uses a balanced 50/50 intervention mixture so
a strict-only gain cannot dominate a collapsed nominal controller; a strict-
only gain without held-out nominal retention is not sufficient.

Before V13 held-out results, a separate V14 extension was frozen to test
whether the matched strict-state policy can supply useful recovery DAgger data.
It retains V13's clean visual initializer, optimizer, 80/20 PPO distribution,
and balanced checkpoint selection, while adding 1.92M matched-seed teacher
interactions. It is eligible only if an independent 768-episode nominal audit
shows the strict teacher retains at least 70% raw and safe success with at most
5% violations. This is privileged training, not pure pixel RL or a
self-supervised-only method, and it cannot revise a primary hypothesis verdict.
Before V14 starts, its training-stream checkpoint-selection cohort is fixed at
256 environments rather than V13's 64 to reduce selection variance observed in
the running diagnostics. This does not change the 768-episode disjoint strict
or nominal held-out endpoints and does not expose their seeds during training.
The teacher subsequently scored 0% nominal raw/safe success with 73.57%
violations, failing every allocation check. V14 and its downstream chain were
therefore cancelled before training. The frozen proposal and gate result remain
negative evidence against distilling a distribution-specialized teacher.

The fair privileged comparator is therefore a fresh integrated-mixture state
PPO, frozen after the teacher rejection and before V13 held-out evaluation. It
matches V13's 80% strict / 20% nominal training distribution, balanced 50/50
checkpoint-selection distribution, seeds, reward, safety terms, and requested
100M-step budget; only the deployed observation modality and vectorization
differ. Strict and nominal final endpoints remain separate. Its 256-environment
training-stream selection cohort reduces checkpoint noise without exposing any
held-out seed.

V15 is a post-registration RGB-student extension frozen before its state
teacher or visual held-out results. It can allocate only if that integrated
state policy passes the final six-endpoint selection thresholds itself. Each
student seed then receives 1.92M same-seed DAgger transitions, starts from the
corresponding V13 RGB checkpoint, and runs 100M further PPO steps on V13's
unchanged distributions. Deployment remains restricted to RGB, robot
proprioception, and instruction. Report any gain as integrated expert
distillation plus PPO under privileged training, never as pure pixel RL or a
self-supervision-only result.

Because both integrated-state teacher routes failed nominal retention, the
released V19 extension instead uses two specialists during training: the
nominal V6 RGB policy and the strict V11 state policy. A training-only physical-
resolution label routes DAgger supervision; the deployed student still receives
only RGB, robot proprioception, and instruction and executes continuous control.
V19 completed its exact 100M-step three-seed audit and frozen held-out suite.
It obtains 96.35% strict safe success (740/768), 91.41% nominal safe success
(702/768), 97.06%/95.69% safe success on the two physical-removal branches,
and 1.30%/3.65% strict/nominal violations. It passes every preregistered
integrated threshold and is the selected restricted-input visual policy at a
91.41% worst endpoint. V20, the exact full-strength VICReg ablation, completed
its audit but failed the same gate: 85.42% strict safe, 90.89% nominal safe,
74.06%/96.19% branch safe, and 2.73%/1.30% strict/nominal violations. Although
V20 improves matched-pixel pose R² by +0.0106 [0.0016, 0.0212] and task-semantic
R² by +0.0146 [0.0010, 0.0377] over V19, its control is worse. Report this as a
representation/control dissociation, not a self-supervised control gain.
Training-stream checkpoint scores must not enter final result tables.

V20's early training diagnostics showed strong seed sensitivity while its
unweighted variance loss made the weighted variance term larger than the
weighted temporal-prediction and value terms. Before any V19/V20 held-out
evaluation, a one-seed V21 stabilization smoke was frozen on the known weak
training seed 9351: it changes only the variance coefficient from 0.01 to 0.001
and caps the diagnostic at 20M steps. Job `1140356` may motivate a later fully
frozen three-seed extension, but the smoke itself supplies no held-out result
and cannot replace V20 in the exact ablation.
Its predeclared allocation gate (`1140357`) requires at least 85% success, at
most 5% violations, and a safety-weighted score at least 0.15 above V20 seed
9351 using only evaluation records at or below the same 20M-step budget. The
gate rejects any unmatched configuration field or incomplete budget and passed
five focused contract tests before submission. The exact three-seed, 100M-step
extension config was also frozen before the smoke result; it matches V20 in all
algorithm and evaluation fields except the variance coefficient and cannot be
allocated unless the gate passes.
The dependency-held full V21 training array, exact checkpoint audit, strict and
nominal held-out evaluations, aggregates, and selector are jobs `1140381`--
`1140387`; they use zero GPU time if gate `1140357` fails. The V21-inclusive
strict table and selection rule were frozen before any V21 held-out result.
Matched-pixel V20/V21 pose probes and task-semantic probes are independently
dependency-held as jobs `1140388`--`1140395`. Their comparators require three
same-seed cohorts, byte-identical RGB datasets and random controls, and report
the configured treatment/control pair; representation decodability remains a
diagnostic rather than evidence of causal control benefit.
Because the treatment/control report label was generalized while the isolated
full suite was active, post-suite delta gate `1140396` hashes every changed
source/config/test file and reruns the 30 affected tests with retained JUnit.
All affected probes and comparators now depend on that gate, avoiding a claim
based on a regression run that silently spans a source edit.
The V21 allocation smoke completed its exact 19,996,672-step budget. Its best
eligible record, at step 18,014,208, reached 90.625% end success, 0.391%
violations, and 0.8986 safety-weighted score, +0.2383 over V20 seed 9351 at the
matched budget. It passed all three preregistered checks and released the
independently frozen three-seed extension. This is a one-seed training-stream
allocation decision, not held-out evidence and not a paper performance result;
the full V21 learning-rate schedule also differs from the smoke schedule's
20M-step annealing horizon.

The matched-pixel V13-versus-V6 pose diagnostic is now complete. Dataset and
behavior-checkpoint hashes match within every seed. V13 improves mean linear
probe R² over V6 by 0.0330, with paired seed-bootstrap interval [0.0141,
0.0619]. However, V6 and V13 learned-feature mean R² values are -0.0399 and
-0.00686 respectively, and neither has a reliably positive learned-minus-
random interval. The defensible statement is a small relative change in
diagnostic decodability, not useful pose recovery, a causal benefit, or proof
that self-supervision explains control performance.

The frozen causal/OOD suite separates three questions: cyclic intervention of
the predicted progress head, image-space perturbations, and renderer-native
camera/lighting changes. Its reset preflight proves the five renderer profiles
produce distinct pixels while preserving byte-identical task/robot/object
state. Only the dependency-held closed-loop evaluation can test robustness.
The primary causal criterion requires a safety-weighted drop of at least 0.03
with a positive paired hierarchical-bootstrap lower bound; the OOD criterion
requires at least 0.75 safe success and no more than a 0.15 upper-CI drop.

V22 is a separately frozen DrAC-style stability ablation, not a modification
to an active V19 run. PPO ratios remain computed from the original image. A
second loss matches the live random-shift policy to a stopped original-image
Gaussian target with exact pre-tanh KL; the asymmetric state critic receives
no visual invariance loss. Accordingly, any eventual claim must say
"DrAC-style policy consistency with an asymmetric critic," not full DrAC.
Runtime, smoke, full training, checkpoint audit, and held-out evaluations were
fail-closed dependencies. The 262,144-step runtime completed all software paths
but fell from 93.75% end success at initialization to 0% at its final
evaluation. The full-DAgger smoke reproduced the failure: unweighted KL reached
1.86e20 and end success was 0% at 0.81M and 1.63M steps. It was stopped at
1.64M, explicitly recorded as ineligible, and no V22 full training or held-out
evaluation was released. A
failure-only, otherwise matched V23 runtime pilot freezes coefficient 0.00009
from a loss-scale cap. Job `1140596` completed at 92.19% end success and 1.56%
violations after starting at 93.75%/3.13%, so it prevents weak-BC collapse.
Raw KL nevertheless remained roughly 1,000--1,400, and V22's full-DAgger KL
was above 1e8; V23 therefore receives no larger allocation. Because the
full-DAgger V22 KL subsequently reached 1.49e20 by 1.39M steps, a separate V24
runtime pilot tests stopped-target Huber consistency on tanh-bounded action
means. Job `1140599` is also failure-only. Its preregistered runtime gate
`1140609` may release only one separately frozen 20M smoke (`1140610`) after
exact completion, bounded finite loss, competence retention, and safety checks;
the runtime passed at 87.50% end success, 4.69% violations, and maximum loss
0.47865. Before the 20M result, a matched best/tail/safety gate (`1140623`) and
three-seed 100M/held-out chain (`1140624`, `1140629`--`1140634`) were frozen.
They allocate only on successive passes and do not convert runtime or smoke
diagnostics into paper performance evidence.

Final extension selection is also frozen before held-out evaluation. A policy
is eligible only with at least 90% strict safe success, 90% nominal safe
success, 85% safe success in each of the first- and second-goal-removal
branches, and no more than 5% violations in either condition. Eligible methods
rank by the minimum of those four safe-success endpoints, not their pooled
average. If every method fails, the selector returns no policy; a strong mean
cannot hide branch collapse or nominal forgetting.

## External comparison language

SayCan, Inner Monologue, REFLECT, RecoveryChaining, Failure-Aware RL,
Sequential Dexterity, SPIRE, DEMO³, MSDP, DrQ-v2, MENTOR, Maniwhere,
CP3ER, masked-modality training, DreamerV3/4, and TD-MPC2
define neighboring problem or algorithm families. Their published scores use
different embodiments, observations, interventions, and success criteria.
Compare capabilities and assumptions in a matrix; do not state that an ATR
percentage exceeds theirs. Competitive evidence comes from strong methods run
under the identical ATR protocol plus transparent sample and privilege costs.

## Abstract/conclusion gate

Do not write “state of the art,” “outperforms prior work,” “robust recovery,” or
“self-supervision is necessary” unless the final strict aggregate, direct
factorial, and five-seed confirmation support the exact phrase. Acceptable
interim language is “a restricted-input visual policy achieves X under the
custom ATR benchmark” with protocol, seeds, intervals, and privilege disclosure.
