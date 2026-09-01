# Adaptive Task Recovery

Adaptive Task Recovery (ATR) studies how a robot should preserve task intent
when the world changes during execution: an object is ejected, a goal becomes
blocked, or a temporary obstruction makes the original plan invalid.

The current method combines temporal evidence accumulation over observed
motion, calibrated abstention, and continuous-control specialists. Runtime
routing is temporally causal -- it reads only observations at or before the
current step -- and excludes intervention identity, future state, oracle
feasibility, and native success labels. "Causal" throughout this repository
refers to that no-future-access property, not to inferring causal dynamics;
see the history ablation below.

<p align="center">
  <img src="media/demos/learned-recovery-montage.gif" width="900" alt="Frozen-policy ManiSkill recovery and nominal-control episodes.">
</p>

## Strongest completed results

Only the latest completed results that support a positive claim are summarized
here. Rejected experiments, development sweeps, and historical results remain
available in the evidence ledger, not in the README.

### Guarded factorized dispatch — untouched confirmation, completed baselines

The V10 candidate was frozen before its once-only `347000000` confirmation
family was opened. Evaluation uses three policy lineages, five recovery
conditions, matched observations and specialists across routers, and a held-out
reverse-ejection mechanism.

The gate's declared method list includes a hand-written V28 heuristic router and
an oracle upper bound. Both were unimplemented when the gate was first scored,
so the original comparison ran three arms rather than five. Both are now built
and evaluated on the same family; the table below is the completed comparison.

The causal arm additionally ran a factorized sweep dispatch that no other arm
can execute. It is reported separately rather than folded in, because only one
arm could receive it.

| Arm | Recurrent | n | Safe | Viol | permanent | temporary | held-out reverse |
|---|:---:|---:|---:|---:|---:|---:|---:|
| **Causal router (matched)** | yes | 2,880 | **88.99%** | **0.83%** | 97.40% | 84.38% | 97.40% |
| Unstructured GRU | yes | 2,880 | 81.74% | 4.83% | 97.40% | 84.20% | 46.88% |
| Hand-written V28 heuristic | no | 960 | 74.06% | 16.98% | **100.0%** | **0.00%** | 97.40% |
| Static offset, one frame | no | 2,880 | 70.90% | 0.83% | **0.00%** | **84.38%** | 97.40% |
| Static MLP, current frame | no | 2,880 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Immediate oracle (privileged) | — | 960 | 89.79% | 1.87% | 99.48% | 76.04% | 90.10% |
| *Causal + factorized dispatch* | yes | *2,880* | *92.19%* | *0.83%* | *97.40%* | *84.38%* | *97.40%* |

Matched-input gains, Newcombe 95%: **+7.26 points [+5.44, +9.07]** over the
unstructured GRU and **+14.93 points [+12.00, +18.00]** over the heuristic. The
causal router is statistically indistinguishable from the privileged immediate
oracle (**−0.80 points [−2.93, +1.55]**) while using no privileged input.

**What actually requires memory.** Mechanism *identification* requires none.
The held-out reverse ejection is solved at an identical **97.40%** by the causal
router, the hand-written threshold rule, and a single-observation MLP with no
sequence encoder at all — three independent methods, one of them memoryless.
Only the unstructured GRU fails it (46.88%). A held-out mechanism that a
one-frame model identifies perfectly cannot support a composition claim.

The discrimination lives entirely in one confusion pair: **permanent versus
temporary obstruction**, where a wrong commitment is unrecoverable. Both
non-recurrent arms fail it in *opposite* directions — the threshold rule calls
everything permanent (100.0% / 0.00%), the one-frame model calls everything
temporary (0.00% / 84.38%). Each solves one side and scores zero on the other.
Both recurrent arms solve both. Against the one-frame model the causal router
gains **+97.40 points [+95.62, +98.42]** on permanent blockage while being
statistically identical on nominal, temporary, and held-out reverse, and
*worse* on forward ejection (−7.12 points).

The supported claim is therefore narrow and mechanistic: **temporal evidence is
required to defer commitment on an obstruction whose persistence is not yet
observable, and for nothing else in this benchmark.**

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

### REBOOT real-robot trajectories — leave-one-object-out transfer

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

Authoritative record:
[`results/a_plus_audit/reboot_v2_aggregate.json`](results/a_plus_audit/reboot_v2_aggregate.json).

### Restricted-RGB continuous recovery

V19 is the strongest completed integrated restricted-input controller. Across
three seeds and 768 held-out episodes per regime it reaches:

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

The completed V10 custom-benchmark result and REBOOT offline transfer are strong
positive evidence, but they are not sufficient for an A/A+ IROS claim. The
independently preregistered, no-teleport ManiSkill `PegInsertionSide-v1`
closed-loop gate is still in progress. Its reserved `425000000` selection and
`429000000` confirmation families remain unopened.

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

The repository records immutable experiment configurations, seed families,
checkpoint hashes, per-episode outcomes, confidence intervals, failed gates,
and Slurm job provenance. Runtime interventions use simulator forces and
contacts; pose assignment is permitted only during reset.

Key entry points:

- [`src/atr/envs/learned_recovery_v4.py`](src/atr/envs/learned_recovery_v4.py):
  mechanism-diverse custom recovery environment.
- [`src/atr/envs/peg_insertion_recovery.py`](src/atr/envs/peg_insertion_recovery.py):
  no-teleport extension of official ManiSkill PegInsertion.
- [`src/atr/policies/causal_option_router.py`](src/atr/policies/causal_option_router.py):
  learned temporally-causal and matched router models.
- [`src/atr/policies/heuristic_option_router.py`](src/atr/policies/heuristic_option_router.py):
  hand-written V28 motion-threshold baseline over the same matched tensor.
- [`scripts/evaluate_external_peg_router.py`](scripts/evaluate_external_peg_router.py):
  matched closed-loop external evaluation.
- [`scripts/summarize_external_peg_gate.py`](scripts/summarize_external_peg_gate.py):
  fail-closed external publication gate.
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

ATR currently establishes simulation closed-loop recovery and offline transfer
on real-robot trajectories. It does **not** claim real-robot closed-loop
recovery, universal visual robustness, or an A/A+ general-method result before
the external PegInsertion gate passes.

Detailed negative results are intentionally retained in [`docs/`](docs/) and
[`results/`](results/) so that the concise README does not become selective
reporting.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
