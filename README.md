# Adaptive Task Recovery

Adaptive Task Recovery (ATR) studies how a robot should preserve task intent
when the world changes during execution: an object is ejected, a goal becomes
blocked, or a temporary obstruction makes the original plan invalid.

The current method combines causal temporal inference, calibrated abstention,
and continuous-control specialists. Runtime routing uses observable trajectory
history; intervention identity, future state, oracle feasibility, and native
success labels are excluded.

<p align="center">
  <img src="media/demos/learned-recovery-montage.gif" width="900" alt="Frozen-policy ManiSkill recovery and nominal-control episodes.">
</p>

## Strongest completed results

Only the latest completed results that support a positive claim are summarized
here. Rejected experiments, development sweeps, and historical results remain
available in the evidence ledger, not in the README.

### Guarded factorized causal dispatch — untouched confirmation

The V10 candidate was frozen before its once-only `347000000` confirmation
family was opened. Evaluation uses three policy lineages, five recovery
conditions, matched observations and specialists across routers, and a held-out
reverse-ejection mechanism.

| Metric | V10 causal dispatch | Strongest matched non-oracle |
|---|---:|---:|
| Safe recovery | **2,655/2,880 — 92.19%** | 2,354/2,880 — 81.74% |
| Gain | **+10.45 points** | — |
| Newcombe 95% interval | **[+8.05, +12.83] points** | — |
| Constraint violations | **24/2,880 — 0.83%** | — |
| Worst condition | **84.38% safe recovery** | — |
| Held-out reverse ejection | **561/576 — 97.40%** | — |

The same frozen controller passed the registered pooled OOD gate at
**6,369/7,680 — 82.93% safe recovery**, above its 75% floor, with **2.81%**
violations. This is a pooled result, not universal robustness: 12-step action
delay reached only 55.83% safe recovery, and a long temporary obstruction
produced 15.83% violations.

Authoritative record:
[`configs/temporal_composition_v10_confirmation.json`](configs/temporal_composition_v10_confirmation.json).

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
control.

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
  learned causal and matched router models.
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
