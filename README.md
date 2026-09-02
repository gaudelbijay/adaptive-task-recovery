# Adaptive Task Recovery

Adaptive Task Recovery (ATR) studies recovery after irreversible world change
during manipulation. Part-way through a multi-goal task an object is ejected, a
goal becomes blocked, or an obstruction appears that may or may not clear. The
agent must determine which goals remain achievable, complete those, and respect
an explicit protected-object constraint.

This repository contains the benchmark, an audit for evaluating whether such a
benchmark measures what it claims, and the experimental record.

## Method: the shortcut ladder

Recovery benchmarks commonly hold out a mechanism that the model never trains
on, then measure whether it composes an appropriate response. A large margin
over a learned baseline is taken as evidence of composition. That inference is
only valid if the held-out mechanism cannot be identified by a model without
the capability under test.

The shortcut ladder scores the held-out mechanism against four controls of
increasing capability, using identical inputs, splits, and targets:

| Rung | Control | Question |
|---|---|---|
| 1 | current frame only | is the mechanism visible instantaneously? |
| 2 | one past frame | is it visible from a single earlier observation? |
| 3 | hand-written rule | is it visible to a motion threshold with no learning? |
| 4 | recurrent model | does identifying it require temporal evidence? |

If a lower rung matches rung 4, the held-out mechanism is a shortcut and no
composition claim follows from it, irrespective of the margin over a weak
baseline.

<p align="center">
  <img src="media/results/shortcut-ladder.png" width="1000" alt="Panel A: held-out mechanism score by control rung for three benchmarks. On LearnedRecovery-v4 a one-past-frame model reaches 1.00, matching the recurrent model, so the held-out mechanism is a shortcut. On PegInsertionSide-v1 no rung below the recurrent model exceeds 0.09. On REBOOT real-robot trajectories the one-past-frame control reaches 0.61 macro-AUROC against 0.80 for the recurrent model. Panel B: closed-loop safe recovery on permanent versus temporary obstruction for four arms; the motion rule scores 1.00 and 0.00, the one-frame model scores 0.00 and 0.84, and both recurrent arms solve both sides.">
</p>

## Results

Applied to three benchmarks with an identical rung set. A lower rung "matches"
rung 4 when the paired bootstrap interval on their difference includes zero,
resampling whole episodes on the simulated benchmarks and object families on
REBOOT:

| Benchmark | Best lower rung | Rung 4 | rung4 − lower | Verdict |
|---|---:|---:|---|:---:|
| `LearnedRecovery-v4` (this work) | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | shortcut |
| `PegInsertionSide-v1` | 0.0909 | 0.4015 | +0.3240 [+0.1344, +0.5231] | none |
| REBOOT (external, real robot, 10 seeds) | 0.7482 | 0.8108 | +0.0626 [+0.0035, +0.1367] | none |

On `LearnedRecovery-v4` two independent non-recurrent controls — a
single past frame and an order-free prefix summary — reach exactly the
recurrent score, so the shortcut does not depend on how one control was built.
On the other two benchmarks every lower rung is separated from the recurrent
model by an interval excluding zero.

The REBOOT margin is thin: the lower bound is +0.0035, and its order-free
summary reaches 0.923 of the recurrent score. An earlier version of this
analysis used a ratio cut at 0.9 and recorded REBOOT as a shortcut on that
basis. The ratio has no error rate attached, so the paired test replaces it;
the two disagree only on REBOOT, and both readings are reported here rather
than only the one the test returns. `LearnedRecovery-v4` and `PegInsertionSide-v1`
are unambiguous under either.

The result on `LearnedRecovery-v4` has a specific cause, and it is structural.
Forward and reverse ejection are produced by *separate actors*, so identifying
the mechanism reduces to noticing which actor moved; a hand-written motion
threshold reaches 97.40% closed-loop doing only that. A second cause compounds
it: current-centering was introduced to remove an earlier shortcut in which
instantaneous geometry identified the mechanism, and it succeeded — rung 1
falls to 0.0322 — but expressing every frame as a signed displacement to the
present made a single early frame carry the whole answer. Removing a leakage
path at one rung opened another at the next.

### Which capability needs memory differs between benchmarks

On `LearnedRecovery-v4`, distinguishing a permanent obstruction from a
temporary one requires temporal evidence: both non-recurrent arms fail the pair
in *opposite* directions while both recurrent arms solve both sides. Closed-loop
on `PegInsertionSide-v1` that reverses — the memoryless arm is the best of the
four on permanent blockage.

| Arm | v4 permanent | v4 temporary | Peg permanent | Peg temporary |
|---|---:|---:|---:|---:|
| Factorized GRU | 0.9740 | 0.8438 | 0.4740 | 0.0052 |
| Unstructured GRU | 0.9740 | 0.8420 | 0.5208 | 0.0156 |
| Static, one frame | 0.0000 | 0.8438 | **0.5677** | 0.0312 |
| Hand-written rule | 1.0000 | 0.0000 | 0.0000 | 0.0365 |

No PegInsertion recovery specialists exist, so the nominal checkpoint filled all
three specialist roles. Every arm shares that handicap, so the comparison
between arms holds, but each scores at or below 0.0365 on temporary blockage:
that column measures the missing specialist rather than the routing, and only
the permanent column supports a conclusion.

### PegInsertion leaks condition identity through episode timing

Peg's features are current-centered, so a model reading only the final frame
receives near-zero geometry. It nonetheless reaches 0.8034 and 0.8424 on the
two blockage conditions. Zeroing `normalized_time` and changing nothing else
removes that advantage entirely:

| Model | Condition | With time | Without | Δ |
|---|---|---:|---:|---:|
| Static MLP | permanent | 0.8034 | 0.6094 | −0.1940 |
| Static MLP | temporary | 0.8424 | 0.6027 | −0.2398 |
| Static MLP | positive ejection | 0.6033 | 0.6071 | +0.0038 |
| Factorized GRU | permanent | 0.7362 | 0.7468 | +0.0106 |

The effect lands on the two blockage conditions and nowhere else, and the
recurrent model is unaffected. This also explains the closed-loop result above:
the memoryless arm won on permanent blockage by reading episode duration rather
than physics. The lesson transfers — mechanisms that terminate episodes
differently produce different clock distributions, so any recovery benchmark
carrying a time feature should ablate it.

### Preregistered gates that failed

Both were frozen and committed before the runs they scored, and neither was
reinterpreted afterwards.

| Gate | Outcome |
|---|---|
| Peg nominal continuation | Failed 4 of 4. Competence regressed to a 0.6862 three-seed mean against a 0.84 baseline, and failures stayed 99–100% single-mode. |
| `LearnedRecovery-v5` physics | Failed 3 of 4. Direction was never established: late separability 0.516 against 0.5 for indistinguishable. The shared-ejector design addressed the right cause; the implementation did not work. |

## Benchmark

<p align="center">
  <img src="media/demos/v4-router-montage.gif" width="900" alt="Three LearnedRecovery-v4 episodes under the frozen router, each labelled with the option the router has committed to. Reverse ejection commits at step 2, temporary blockage at step 38, permanent blockage at step 52. All three are verified safe successes.">
</p>

<p align="center"><sub>
The frozen router on <code>LearnedRecovery-v4</code>, one panel per mechanism,
annotated with the option selected at each step. It commits to reverse ejection
at step 2, and observes for 38 and 52 steps respectively before committing on
temporary and permanent blockage. All three episodes are verified safe
successes; capture provenance is in <code>results/v4_capture/</code>.
</sub></p>

The deferral asymmetry is the behavioural counterpart of the result above:
whether an obstruction will clear is not observable when it first appears.

Earlier footage of the restricted-RGB controller on `LearnedRecovery-v3` is
retained in
[`media/demos/learned-recovery-montage.gif`](media/demos/learned-recovery-montage.gif).
That environment contains no blockage mechanisms and does not illustrate this
behaviour.

Runtime routing is temporally causal — it reads only observations at or before
the current step — and excludes intervention identity, future state, oracle
feasibility, and native success labels. "Causal" throughout this repository
refers to that no-future-access property, **not** to inferring causal dynamics;
the history ablation below rejects the stronger reading.

## Detailed results

The measurements behind the summary above, with the complete baseline set.
Rejected experiments, development sweeps, and historical results are recorded
in the evidence ledger rather than here.

### Closed-loop router comparison

The guarded factorized dispatch router was frozen before its once-only
`347000000` confirmation family was opened. Evaluation uses three policy
lineages, five recovery
conditions, matched observations and specialists across routers, and a held-out
reverse-ejection mechanism.

The gate's declared method list includes a hand-written motion-threshold router
and an oracle upper bound. Both were unimplemented when the gate was first scored,
so the original comparison ran three arms rather than five. Both are now built
and evaluated on the same family; the table below is the completed comparison.

The recurrent factorized arm additionally ran a factorized sweep dispatch that
no other arm can execute. It is reported separately rather than combined,
because only one arm could receive it.

| Arm | Recurrent | n | Safe | Viol | permanent | temporary | held-out reverse |
|---|:---:|---:|---:|---:|---:|---:|---:|
| Factorized router (matched inputs) | yes | 2,880 | 88.99% | 0.83% | 97.40% | 84.38% | 97.40% |
| Unstructured GRU | yes | 2,880 | 81.74% | 4.83% | 97.40% | 84.20% | 46.88% |
| Hand-written motion rule | no | 960 | 74.06% | 16.98% | 100.00% | 0.00% | 97.40% |
| Static offset, one frame | no | 2,880 | 70.90% | 0.83% | 0.00% | 84.38% | 97.40% |
| Static MLP, current frame | no | 2,880 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Immediate oracle (privileged) | — | 960 | 89.79% | 1.87% | 99.48% | 76.04% | 90.10% |
| Factorized router + sweep dispatch | yes | 2,880 | 92.19% | 0.83% | 97.40% | 84.38% | 97.40% |

Matched-input differences, Newcombe 95% intervals: 7.26 points [5.44, 9.07]
over the unstructured GRU and 14.93 points [12.00, 18.00] over the hand-written
rule. The factorized router is statistically indistinguishable from the
privileged immediate oracle at −0.80 points [−2.93, 1.55], while receiving no
privileged input.

Mechanism identification does not require memory. The held-out reverse ejection
is solved at 97.40% by the factorized router, the hand-written threshold rule,
and a single-observation model alike; only the unstructured GRU fails it, at
46.88%. A held-out mechanism that a one-frame model identifies perfectly cannot
support a composition claim.

The discrimination occurs in one condition pair, permanent versus temporary
obstruction, where an incorrect early commitment is unrecoverable. The two
non-recurrent arms fail this pair in opposite directions: the threshold rule
commits to permanent (100.00% / 0.00%) and the one-frame model to temporary
(0.00% / 84.38%). Each solves one side and scores zero on the other, while both
recurrent arms solve both. Against the one-frame model the factorized router
gains 97.40 points [95.62, 98.42] on permanent blockage, is statistically
indistinguishable on nominal, temporary, and held-out reverse, and is 7.12
points lower on forward ejection.

The supported claim is therefore restricted, and restricted further by the
cross-benchmark result above: on *this* benchmark temporal evidence is required
to defer commitment on an obstruction whose persistence is not yet observable,
and is not required elsewhere in it. That does not hold on PegInsertion, where
the memoryless arm is the best of the four on permanent blockage.

The static MLP's 0.00% is structural, not a defeated baseline: current-centering
makes the final geometry frame exactly zero (audited, `final_geometry_max_abs =
0.0`), so a model that sees only that frame receives an all-zero input. The
`static offset` arm replaces it with a real single-observation control that
reads one earlier frame; offline it reaches **100%** held-out reverse accuracy,
selected across 16-, 48-, and earliest-frame offsets by validation only.

Two ablations constrain the mechanism further. Reversing the prefix in time
leaves held-out accuracy at **97.7% / 77.6% / 96.9%**, so temporal *direction*
is not used. And stripping history from a history-trained model drops it to
**0.000** — but that measures degradation under distribution shift, not
necessity, since a model *trained* on one frame reaches 100%. The supported
mechanism is temporal aggregation of signed motion evidence with deferred
commitment, not causal dynamics inference.

The same frozen controller passed the registered pooled OOD gate at
**6,369/7,680 — 82.93% safe recovery**, above its 75% floor, with **2.81%**
violations. This is a pooled result, not universal robustness: 12-step action
delay reached only 55.83% safe recovery, and a long temporary obstruction
produced 15.83% violations.

Authoritative records:
[`configs/temporal_composition_v10_confirmation.json`](configs/temporal_composition_v10_confirmation.json)
and [`results/router/matched_router_comparison_347M.json`](results/router/matched_router_comparison_347M.json).
The completed five-arm comparison and the history ablations are development
evidence on an already-opened family, not a second gate pass.

### REBOOT: leave-one-object-out transfer on real-robot trajectories

The causal recovery-state predictor was evaluated offline on **2,072 usable
real-robot trajectories** from the 2026
[REBOOT benchmark](https://nanayawoa.github.io/REBOOT/). All models use matched
inputs and leave each of nine object families out in turn.

| Recovery-state predictor | Macro-AUROC |
|---|---:|
| Static MLP | 0.5797 |
| Trajectory-moment MLP | 0.7450 |
| Unstructured GRU | 0.8072 |
| **Causal dynamics GRU** | **0.8353** |

The causal model improves over the static model by **+25.56 AUROC points**
(object-bootstrap 95% interval **[+21.10, +29.47]**) and over the
trajectory-moment baseline by **+9.03 points** (**[+1.96, +18.08]**). Its
+2.82-point difference from the unstructured GRU is not statistically resolved
(**[−1.40, +9.38]**), so no architecture-superiority claim is made for that
comparison. This is real-robot offline inference, not real-robot closed-loop
control. The model name follows the simulator router's; the history-direction
ablation above was run on that router, not on this offline predictor, so no
causal-dynamics claim is made here either.

The control ladder was also run on this benchmark, where it serves as the
audit's negative control: the endpoint-pair rung reaches 0.6080 against 0.8045
for the recurrent model, a difference of 0.1966 [0.1557, 0.2357]. This
benchmark's held-out family requires temporal structure.

Adding that rung exposed an order-dependent seeding defect: model weights were
initialised from the advancing global RNG, so inserting a method silently
re-initialised every method after it. `fit_one` now seeds per method and fold.
With that fixed the causal-versus-unstructured comparison is **−0.0067 [−0.0249,
+0.0119]**, still unresolved, consistent with the original run.

Authoritative records:
[`results/a_plus_audit/reboot_v2_aggregate.json`](results/a_plus_audit/reboot_v2_aggregate.json)
and [`results/a_plus_audit/reboot_ladder_v4_aggregate.json`](results/a_plus_audit/reboot_ladder_v4_aggregate.json).

### Restricted-RGB continuous control

The dual-specialist RGB controller is the strongest completed integrated
restricted-input policy. Across three seeds and 768 held-out episodes per
regime it reaches:

| Regime | Safe success | Violations |
|---|---:|---:|
| Strict physical removal | **96.35%** | 1.30% |
| Nominal two-goal task | **91.41%** | 3.65% |
| First goal removed | **97.06%** | included above |
| Second goal removed | **95.69%** | included above |

The actor executes continuous joint control from restricted RGB, robot
proprioception/TCP, the instruction, and learned progress. Object poses,
intervention labels, and evaluator domains are unavailable to the deployed
actor. Training uses privileged teachers and labels, so this is not a pure
self-supervised or end-to-end pixel-RL claim.

Evidence and claim boundaries:
[`docs/14-results-and-claim-boundaries.md`](docs/14-results-and-claim-boundaries.md)
and [`docs/17-visual-training-ledger.md`](docs/17-visual-training-ledger.md).

## Publication status

The independently preregistered, no-teleport ManiSkill `PegInsertionSide-v1`
closed-loop experiment is still in progress. Its reserved `425000000` selection
and `429000000` confirmation families remain unopened.

No running Peg result is reported above. Promotion requires all of the
following without relaxing thresholds after seeing outcomes:

1. three-seed official PegInsertion nominal competence;
2. input-matched causal routing that passes held-out-direction composition;
3. stronger closed-loop performance than static, unstructured recurrent, and
   heuristic routers with the same observations and specialists;
4. at least 80% safe recovery, at least 75% held-out-direction recovery, no
   more than 3% violations, and a significant gain of at least five points;
5. a once-only untouched `429000000` confirmation.

The complete protocol is in
[`docs/19-iros-publishability-gate.md`](docs/19-iros-publishability-gate.md)
and [`docs/30-a-plus-recovery-protocol.md`](docs/30-a-plus-recovery-protocol.md).

## Reproducibility

**Naming.** Prose here names methods by what they do. Version tags survive only
where they identify a frozen artifact — a registered environment
(`LearnedRecovery-v4`),
a preregistered gate config, a checkpoint's persisted `model` string, or a
reserved seed family. Those are immutable identifiers with provenance attached,
not a quality ordering: a higher number is a later candidate, not a better one,
and most were rejected. The per-candidate ledger lives in
[`docs/30-a-plus-recovery-protocol.md`](docs/30-a-plus-recovery-protocol.md).

The repository records immutable experiment configurations, seed families,
checkpoint hashes, per-episode outcomes, confidence intervals, failed gates,
and Slurm job provenance. Runtime interventions use simulator forces and
contacts; pose assignment is permitted only during reset.

Key entry points:

- [`src/atr/envs/learned_recovery_v4.py`](src/atr/envs/learned_recovery_v4.py):
  mechanism-diverse custom recovery environment.
- [`src/atr/envs/peg_insertion_recovery.py`](src/atr/envs/peg_insertion_recovery.py):
  no-teleport extension of official ManiSkill PegInsertion.
- [`src/atr/policies/option_router.py`](src/atr/policies/option_router.py):
  learned factorized and matched-baseline router models.
- [`src/atr/policies/heuristic_option_router.py`](src/atr/policies/heuristic_option_router.py):
  hand-written motion-threshold baseline over the same matched tensor.
- [`scripts/evaluate_external_peg_router.py`](scripts/evaluate_external_peg_router.py):
  matched closed-loop external evaluation.
- [`scripts/summarize_external_peg_gate.py`](scripts/summarize_external_peg_gate.py):
  fail-closed external publication gate.
- [`scripts/audit_shortcut_ladder.py`](scripts/audit_shortcut_ladder.py): the
  four-rung shortcut audit, and
  [`scripts/plot_shortcut_ladder.py`](scripts/plot_shortcut_ladder.py) which
  renders the figure above from committed artifacts.
- [`docs/18-evidence-blueprint.md`](docs/18-evidence-blueprint.md): evidence
  provenance and claim boundaries.

Install the project in a Python environment with its simulator dependencies,
then run the non-GPU contract tests:

```bash
pip install -e .
pytest -q
```

GPU experiments are designed for Slurm and use the checked-in launch scripts.
Do not substitute development seeds for reserved confirmation families.

## Scope

ATR establishes closed-loop recovery in simulation and offline transfer on
real-robot trajectories. All closed-loop control is simulated; the real-robot
evidence is offline inference on recorded trajectories. Visual robustness holds
within the declared camera and lighting distribution and degrades outside it.

Three further limits bound what the results above license. The benchmark the
audit flags is an easy manipulation task: 5 cm cubes onto 9 cm pads with a 4 cm
tolerance and no orientation requirement, a primitive this repository's own
PickCube policy solves at 98.31%. Its interventions are scripted exogenous
events fired at step 0, so recovery here means recognising that a goal is gone
and completing the other, which is goal filtering rather than recovery from
execution failure. An attempt to replace them with emergent failures found the
nominal controller's own failures to be 98.8% a single mode, so that route is
not currently open.

The audit is the contribution. The factorized architecture is reported as a
disclosed negative: statistically indistinguishable from a capacity-matched
plain GRU on REBOOT at −0.0021 [−0.0123, +0.0069] over ten optimizer seeds, and
reaching 0.0199 on genuinely observed held-out prefixes on PegInsertion.

Detailed negative results are intentionally retained in [`docs/`](docs/) and
[`results/`](results/) so that the concise README does not become selective
reporting.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
