# recovery audit protocol (frozen before development)

> **Naming.** Counters in this document belong to the *recovery router* series
> unless marked otherwise; V19 and V28 are *visual*-series artifacts. A counter
> is not a quality ordering. See
> [`31-naming-and-identifier-key.md`](31-naming-and-identifier-key.md).


This protocol replaces the previous goal of improving the hand-written hybrid controller's aggregate.  That controller
is a baseline: its routing logic is a hand-written motion-threshold state
machine, and its primary comparison does not match the router and policy input
contract.  Those facts prevent it from supporting a top-tier learning claim.

The candidate method is a causal recurrent option router.  At time `t` it may
use only observations at or before `t`; evaluator mechanism IDs, intervention
targets, future frames, and `critic_goal_resolved` are forbidden.  It predicts
calibrated values for the same frozen recovery options available to every
matched baseline and may abstain while the posterior is not decisive.  Future
physical persistence and realized option outcomes are training signals only.

Each per-step input is instantaneous task-relative geometry, current robot
state, instruction/progress, and time.  Reset displacement, finite-difference
velocity, and other hand-engineered history summaries are excluded, so the
static baseline cannot receive recurrence by proxy. Prefix timestamps are
pre-action in both collection and deployment. The 96-step causal horizon spans
the longest delayed-onset plus temporary-return window. Before a delayed event
is physically observable, its target is nominal execution—not abstention or a
future mechanism label.

The exact numerical gate, seed families, conditions, baselines, OOD axes, and
anti-shortcut audits are frozen in
`configs/a_plus_recovery_gate_v1.json`.  Development and selection results may
change the method, but not the confirmation seeds or pass thresholds.  A new
candidate starts a fresh selection run.  The confirmation bank is opened once,
after the implementation and calibration are frozen.

## Evidence hierarchy

1. The primary result is closed-loop safe success in LearnedRecovery-v4, with
   all learned and heuristic primary baselines receiving the same state input.
2. Mechanism, timing, force, control-delay, and renderer shifts are reported
   separately and pooled only after every manifest is complete.
3. REBOOT is an external real-robot *offline transfer* benchmark.  It tests
   causal recovery-state prediction from real bimanual trajectory prefixes;
   it is not described as real-robot closed-loop control.
4. The dual-specialist RGB controller (visual series V19) remains context evidence and is never presented as an
   input-matched primary baseline for a state-observation router.

## Stopping rule

The README headline changes only if every frozen criterion passes on the
untouched confirmation bank.  Otherwise the repository records the candidate
as rejected, including the failed condition and uncertainty interval.  No
subset, seed, or favorable OOD profile can substitute for the frozen gate.

## V2 temporal-composition addendum

The V1 (the instantaneous-geometry router) audit found that the five simulator mechanisms are almost perfectly
identified from one task-relative state: the causal and unstructured GRUs
produced the same 878/960 safe successes and the static MLP was one episode
behind. V1 remains rejected. Further tuning on that representation cannot
support a causal-memory claim.

Before V2 (the temporal-composition router) training or evaluation, the follow-up protocol is frozen in
`configs/a_plus_recovery_gate_v2_temporal_composition.json`. At decision time,
the current 42-dimensional task geometry is subtracted from every geometry
frame in the prefix. This is deployably causal and translation-invariant: the
final geometry is exactly zero, while earlier frames retain only motion
relative to the present. Every method receives the identical transformed
tensor. The reverse option is withheld from option cross-entropy; factorized
event and physical-direction supervision remains available to both the causal
and static structured models. Thus causal memory and physical factorization
must both work to compose the held-out option. The unstructured GRU tests
memory without the compositional mapping, and the static factorized MLP tests
the mapping without history.

The V2 selection family is `327000000`. The `331000000` confirmation family is
untouched until all code, checkpoints, thresholds, and specialists are frozen.
Passing V2 does not retroactively make V1 pass.

### V2 rejection and V3 feature-contract correction

V2 (the temporal-composition router) is rejected before closed-loop evaluation. Across three optimizer seeds,
the causal factorized router reached 100% held-out reverse accuracy and the
unstructured GRU reached 0%, but the static factorized MLP reached 66.11%,
above the frozen 40% shortcut ceiling. Inspection of the mechanically named
feature contract found that V2 centered the first 42 actor/mechanism geometry
dimensions but omitted dimensions 42:57, which encode TCP position relative
to cubes, goals, and the protected object.

V3 (the full-geometry centered router) is frozen in `configs/a_plus_recovery_gate_v3_full_geometry.json` before
training. It centers the complete 57-dimensional geometry prefix and changes
no numerical threshold. Its selection family is `328000000`; its untouched
confirmation family is `332000000`. V2 remains a machine-recorded failed
candidate and is not pooled with V3.

### V3 confirmation rejection and V4 nominal-controller correction

V3 (the full-geometry centered router) was opened once on its untouched `332000000` confirmation family. The
causal router achieved 2655/2880 (92.19%) safe successes with 51/2880 (1.77%)
violations, versus 2369/2880 (82.26%) for the strongest non-oracle baseline.
The 9.93-point gain had a 95% Newcombe interval of [7.54, 12.29] points. It
also achieved 573/576 safe successes on the held-out reverse condition.
Nevertheless, nominal safe success was only 456/576 (79.17%), below the frozen
82% worst-condition floor. V3 is therefore rejected and is never rerun as an
untouched result.

V4 (the nominal-state revision) is preregistered in `configs/a_plus_recovery_gate_v4_nominal_state.json`.
It changes only the controller shared by nominal execution and temporary
recovery after clearance: the dual-specialist RGB controller is replaced with a dedicated
LearnedRecovery-v4 state PPO trained on nominal episodes. The exact selected
checkpoint is shared by every router baseline. The 57-dimensional temporal
representation, specialists, router calibration, 36-step safe hold, OOD axes,
and every numerical threshold remain unchanged. V4 uses `329000000` for
selection and reserves `333000000` for a once-only untouched confirmation.

### V4 selection rejection and V5 controller selection

V4 (the nominal-state revision) is rejected before confirmation. Two of its three state PPO seeds learned
0% nominal success; the best seed's frozen checkpoint achieved only 100/192
(52.08%) safe successes on the `329000000` nominal selection episodes. There
were zero violations, but the 82% condition floor failed decisively. The
reserved `333000000` family remains untouched.

The same selection family was used for a declared five-way screen of the
existing shared nominal controller. That controller's seed 9351 achieved 166/192 (86.46%)
safe successes, compared with 135/192 and 152/192 for the other individual
seeds, 150/192 for the mean ensemble, and 145/192 for the median ensemble. V5 (the selected-nominal router)
freezes seed 9351's exact checkpoint and hash in
`configs/a_plus_recovery_gate_v5_selected_nominal.json`. It changes no router,
specialist, representation, hold, or numerical threshold, and uses fresh
`330000000` selection and untouched `334000000` confirmation families.

### V5 confirmation rejection and V6 option-specific library

V5 (the selected-nominal router) was opened once on `334000000`. It achieved 2623/2880 (91.08%) safe
successes and beat the matched unstructured model by 7.57 points with a 95%
Newcombe interval of [5.16, 9.96] points. It is nevertheless rejected: 87/2880
violations is 3.0208%, one episode beyond the 3.00% cap, and temporary recovery
was 468/576 (81.25%), five episodes below the 82% condition floor. These are
not rounded into passes, and no V5 OOD confirmation is run.

V6 (the option-specific nominal router) freezes an option-specific shared controller library in
`configs/a_plus_recovery_gate_v6_option_specific_nominal.json`. Nominal option
0 keeps V5's selected seed-9351 controller; temporary post-clearance option 4
uses the seed-1788 controller that was stronger on that recovery role. The
library is identical for every router. V6 changes no routing model, specialist,
feature, calibration, hold, or numerical criterion. It uses `335000000` for
selection and reserves `339000000` for once-only confirmation.

### V6 selection rejection and V7 evidence-conditioned defer

V6 (the option-specific nominal router) is rejected before confirmation. It passed overall safe success, violation,
gain, uncertainty, held-out reverse, and temporary recovery, but nominal safe
success fell to 435/576 (75.52%) on the fresh `335000000` selection family.
The reserved `339000000` family remains untouched.

Development on the opened selection family showed that the universal 36-step
defer, not recovery routing, was placing the nominal controller outside its
training-state distribution. V7 (the evidence-release router) retains 36 as a maximum hold but releases it
when the method's own calibrated posterior confirms nominal execution. Later
events remain revisable. Every method receives the same release rule; no
mechanism label is consulted. A declared controller screen chose its seed 1788,
which achieved 167/192 (86.98%) nominal safe successes and 893/960 (93.02%)
across all conditions in the causal development run. V7 is frozen for fresh
`336000000` selection and untouched `340000000` confirmation in
`configs/a_plus_recovery_gate_v7_evidence_release.json`.

### V7 selection rejection and V8 natural termination

V7 (the evidence-release router) is rejected before confirmation. It passed overall success, violation,
gain, uncertainty, held-out reverse, and every recovery condition, but nominal
safe success was 464/576 (80.56%) on `336000000`. Its reserved `340000000`
confirmation family remains untouched.

Nine nominal episodes had reached native success and then violated only because
the fixed-size vector evaluator continued issuing and scoring actions after the
environment's natural terminal boundary. V8 (the terminal-ensemble router) freezes first-resolution scoring:
an episode ends at its first success or constraint violation, simultaneous
success/violation remains unsafe, and later masked simulator steps are not
scored. This rule is identical for all methods and does not provide a router
input. V8 also freezes the predeclared mean of all three nominal controllers;
temporary option 4 remains seed 1788. On opened development data this reached
886/960 (92.29%) safe successes, 10 violations, and 162/192 (84.38%) in each
weakest condition. `configs/a_plus_recovery_gate_v8_terminal_ensemble.json`
uses fresh `337000000` selection and untouched `341000000` confirmation.

### V8 selection rejection and V9 reverse-handoff specialist

V8 (the terminal-ensemble router) is rejected before confirmation. It reached 2644/2880 (91.81%) causal safe
successes with 43/2880 (1.49%) violations; its weakest conditions were
492/576 (85.42%), and held-out reverse was 537/576 (93.23%). The strongest
non-oracle unstructured GRU reached 2514/2880 (87.29%). The causal gain was
therefore 4.51 points with a 95% Newcombe interval of [2.29, 6.73] points: a
positive result, but below the unchanged five-point magnitude floor. The
reserved `341000000` family remains untouched.

The causal router's final reverse decision accuracy was 100% in every V8
reverse manifest; residual failures were downstream controller failures.
On the now-open `337000000` development family, a screen of existing reverse
controllers rejected action averaging (144/192) and selected the seed-4796
handoff checkpoint (188/192 safe, zero violations). V9 (the reverse-handoff router) changes only this exact
option-2 checkpoint, shared by every router, and freezes its SHA-256 in
`configs/a_plus_recovery_gate_v9_reverse_handoff.json`. Fresh selection is
`338000000`; `342000000` is reserved for once-only confirmation.

### V9 OOD rejection and V10 guarded factorized dispatch

V9 (the reverse-handoff router) passed every primary selection check at 2647/2880 (91.91%) safe success,
31/2880 (1.08%) violations, and a +5.49-point gain over unstructured with a
95% Newcombe interval of [3.23, 7.73] points. It is nevertheless rejected
before confirmation because pooled OOD safe success was 5348/7680 (69.64%),
below 75%. Delayed onset produced only 330/960 and 383/960 safe successes and
roughly 48% violations. The reserved `342000000` family remains untouched.

On the opened OOD family, a privileged-timing oracle that waited until the
first causally observable post-event prefix established that the delayed tasks
were recoverable. The failure decomposed into conservative joint dispatch and
a forward specialist tied to one handoff distribution. V10 (the guarded factorized dispatch router) therefore freezes
three changes: group-disjoint 99%-precision calibration of the existing causal
event/direction heads, dense pre-action queries under a label-free guard through
step 40, and an existing seed-84293 forward specialist that achieved 181/192,
181/192, and 180/192 safe ejection recoveries at onset 0/12/36 with zero
violations. Every router shares the guard and controllers; only models with a
validated factorized dispatch may use it. Fresh selection is `343000000`, and
`347000000` is reserved for untouched confirmation.

### V10 untouched confirmation passes; external gate remains blocking

The V10 (the guarded factorized dispatch router) candidate was frozen and committed before `347000000` was opened. Its
once-only primary confirmation passed every preregistered check: 2655/2880
(92.19%) safe successes, 24/2880 (0.83%) violations, 84.38% in the worst
condition, and 561/576 (97.40%) on held-out reverse ejection. The strongest
matched non-oracle baseline was the unstructured GRU at 2354/2880 (81.74%).
The causal gain was 10.45 points with a 95% Newcombe interval of [8.05, 12.83]
points.

The same frozen candidate passed the registered pooled OOD confirmation at
6369/7680 (82.93%) safe successes, above the 75% floor. Axis-level results are
not hidden: four-step and twelve-step control delay reached only 73.33% and
55.83%, and the 48-step temporary-block axis incurred 15.83% violations. Thus
V10 establishes the pooled result, not universal delay robustness. The exact
machine-readable record is
`configs/temporal_composition_v10_confirmation.json`.

This closes the input-matched positive-method blocker on LearnedRecovery-v4.
It does not close the evidence standard. The independently preregistered, no-teleport
`PegInsertionSide-v1` closed-loop experiment must still pass before a
general recovery claim or a change to the README headline.

### V10 completed method list and mechanism attribution

The gate's `methods` list declared five arms. When V10 (the guarded factorized dispatch router) was first scored, only
three existed in code: `heuristic_v28_router` and
`oracle_mechanism_router_upper_bound` were names with no implementation, and
`scripts/evaluate_v4_learned_option_router.py` dispatched `causal_gru`,
`static_mlp`, and `unstructured_gru` only. The declared comparison was
therefore incomplete, and this is recorded rather than quietly corrected.

Both missing arms are now implemented
(`src/atr/policies/heuristic_option_router.py`, twelve contract tests) and
evaluated on the same `347000000` family with matched observations,
specialists, seeds, and execution settings. Because that family is already
opened, all results in this section are development evidence and cannot
substitute for the original once-only confirmation.

Two unmatched factors were also isolated:

1. The causal arm ran a factorized sweep dispatch that no other arm can
   execute (`"factorized_sweep_dispatch": "causal only"`). Disabling it changes
   only forward ejection, 96.88% to 80.90%, and leaves held-out reverse
   ejection at exactly 561/576 either way. The matched pooled result is
   88.99%, and the matched gain over the unstructured GRU is 7.26 points
   [5.44, 9.07] -- still above the frozen five-point floor without the
   unmatched mechanism.
2. The static MLP scores 0.00% by construction. Current-centering forces the
   final geometry frame to exactly zero (`final_geometry_max_abs = 0.0`), so a
   model reading only that frame receives an all-zero input. The declared
   factorial cannot separate "history is required" from "this arm was handed a
   zero vector"; a static arm trained on a non-final frame would be needed.

The completed comparison relocates the finding. The hand-written heuristic
solves four of five mechanisms and matches the causal router exactly on the
held-out reverse ejection (both 97.40%), so held-out composition is not
evidence of learned causal structure -- a motion threshold achieves it. On
forward ejection and permanent blockage the heuristic is slightly better. The
learned router's advantage is concentrated entirely in temporary-versus-
permanent obstruction, 84.38% against 0.00% (+84.38 points [80.63, 87.11]),
and in constraint violations, 0.83% against 16.98%. Pooled, the causal router
is statistically indistinguishable from a privileged immediate oracle
(-0.80 points [-2.93, 1.55]) while consuming no privileged input.

The preregistered history ablations were also unrun and are now complete on
4,544 held-out reverse prefixes per seed. Removing geometry history collapses
held-out accuracy to 0.000 on all three seeds. Reversing the prefix leaves it
at 97.7%, 77.6%, and 96.9%. History is necessary; its temporal direction
largely is not. Claims must therefore say temporal aggregation of signed
motion evidence, not causal dynamics inference.

Machine-readable records: `results/router/matched_router_comparison_347M.json`
and `results/router/v18_factorized_dispatch/history_ablation_seed{0,1,2}.json`.

### Terminology: two senses of "causal"

The word carries two meanings in this repository and only one is supported by
evidence.

*Temporally causal* means the model reads no observation after the current
step. `current_centered_sequence` centers on the current frame, prefix
timestamps are pre-action in both collection and deployment, and
`causal_safe_targets` builds targets without future events. This property is
audited and holds.

*Causal-dynamics inference* would mean the model recovers how the intervention
evolved. The history-direction ablation rejects this reading: reversing the
prefix leaves held-out reverse accuracy at 97.7%, 77.6%, and 96.9% across
seeds, while removing history collapses it to 0.000. Under current-centering
each frame already carries a signed displacement to the present, so direction
is available per frame and the model aggregates rather than integrates
dynamics.

Python identifiers have since been renamed to match (see "Rename" below);
only the persisted checkpoint `model` string stays frozen, because gate
configs, `router_checkpoint_sha256` provenance, and result manifests key on it.
Prose must state the supported mechanism: temporal aggregation of signed motion
evidence, with commitment deferred until persistence is established.

### Single-observation baseline that is not handed zeros

`StaticOptionRouter` reads the final frame, which centering forces to exactly
zero, so its 0.00% is structural. The declared factorial therefore cannot
separate "history is required" from "this arm received no input".
`StaticOffsetRouter` closes that gap: it reads one *earlier* frame, which under
centering carries the signed displacement between then and now. It is a genuine
single-observation model with real information and no sequence encoder.
Variants are trained at the earliest valid frame and at 16- and 48-step
offsets, capacity-matched at hidden dimension 96, on the same data, seeds, and
held-out option. Selection among offsets uses group-disjoint validation only.

### Completed arm set: what memory is actually for

The single-observation control resolves the question the original factorial
could not. Offline, `static_offset_first` reaches 100% held-out reverse
accuracy with no sequence encoder; the 16- and 48-step offsets reach 55.72% and
82.17%, a monotone ordering in how far back the frame sits. Selection used
group-disjoint validation only (97.43% / 94.57% / 84.30%).

Closed-loop on `347000000`, per condition:

| Arm | Recurrent | permanent | temporary | held-out reverse |
|---|---|---:|---:|---:|
| causal GRU (matched) | yes | 97.40% | 84.38% | 97.40% |
| unstructured GRU | yes | 97.40% | 84.20% | 46.88% |
| hand-written motion rule | no | 100.00% | 0.00% | 97.40% |
| static offset, one frame | no | 0.00% | 84.38% | 97.40% |

Two conclusions follow, and both narrow the earlier claim.

Mechanism identification requires no memory. Three independent methods reach an
identical 97.40% on the held-out reverse ejection, one of them a memoryless
MLP. The 50.52-point margin over the unstructured GRU measures that arm's
single-option head failing at something otherwise trivial, not learned causal
composition. No composition claim should be made from this benchmark.

Memory is required for exactly one thing: the permanent/temporary confusion
pair, where committing early to the wrong side is unrecoverable. Both
non-recurrent arms fail it in opposite directions -- the threshold rule commits
to permanent and scores 0.00% on temporary, the one-frame model commits to
temporary and scores 0.00% on permanent. Both recurrent arms solve both sides.
Against the one-frame control the causal router gains 97.40 points
[95.62, 98.42] on permanent blockage, is statistically identical on nominal,
temporary, and held-out reverse, and is 7.12 points worse on forward ejection.

The defensible claim is therefore: temporal evidence is required to defer
commitment on an obstruction whose persistence is not yet observable, and for
nothing else this benchmark measures.

This also revises the reading of the history ablation. Stripping history from a
history-trained model dropping to 0.000 measures degradation under distribution
shift, not necessity: a model trained on a single frame reaches 100%. Neither
that ablation nor the static MLP's structural 0.00% was evidence that the task
requires memory.

### Rename: identifiers now describe the model, strings still identify artifacts

`causal_option_router.py` is now `option_router.py`, `CausalOptionRouter` is
`FactorizedOptionRouter`, and `causal_safe_targets` is
`deployable_option_targets`. The old names asserted causal-dynamics inference
that the reversed-prefix and single-frame results do not support.

The persisted `model` string inside a checkpoint is deliberately *not* renamed
and remains `"causal_gru"`. Frozen gate configs, the
`router_checkpoint_sha256` block of
`configs/temporal_composition_v10_frozen_candidate.json`, every result
manifest, and `scripts/audit_temporal_composition_router.py` all key on that
exact string. Renaming it would either invalidate the once-only confirmation's
provenance or require rewriting immutable records. Python identifiers therefore
describe what the model is; the persisted string identifies a specific frozen
artifact, and the two are allowed to differ. Both dispatch sites carry a
comment stating this.

Verified after the rename: all three frozen V10 (the guarded factorized dispatch router) checkpoints load under the new
class names with unchanged SHA-256 digests, including
`causal_seed0 = 532b763e...`, and 79 contract tests pass.

### The shortcut ladder, applied to both benchmarks

`scripts/audit_shortcut_ladder.py` scores a benchmark's held-out mechanism
against four rungs of increasing capability on the identical matched tensor,
group-disjoint split, and held-out option: the current frame only
(instantaneous), one earlier frame (no sequence model), a hand-written
motion-threshold rule, and the recurrent models. A held-out mechanism that a
lower rung identifies as well as the top rung is a shortcut and cannot support
a composition claim.

Held-out-option accuracy, three seeds where applicable:

| Rung | Control | LearnedRecovery-v4 | PegInsertion (cf) | PegInsertion (real) |
|---|---|---:|---:|---:|
| 1 | instantaneous | 0.0322 | 0.0000 | 0.0000 |
| 2 | one past frame | **1.0000** | 0.0909 | 0.0104 |
| 3 | hand-written | 0.1195 | 0.0000 | 0.0000 |
| 4 | recurrent factorized | **1.0000** | 0.4015 | 0.0199 |
| 4 | recurrent unstructured | 0.0000 | 0.0000 | 0.0000 |
| | **shortcut detected** | **yes** | no | no |

The two benchmarks return opposite verdicts, which is what validates the audit
rather than merely applying it.

`LearnedRecovery-v4`'s held-out reverse ejection is a shortcut. A
single-observation model with no sequence encoder matches the recurrent model
exactly at 1.0000. The 50.52-point closed-loop margin over the unstructured GRU
therefore measures that arm's failure at something a memoryless model does
perfectly, not learned composition.

`PegInsertionSide-v1`'s held-out negative ejection is not. No rung below the
recurrent model exceeds 0.0909, and on genuinely observed negative-ejection
prefixes every method including the factorized router is at or below 0.0199.
That benchmark's held-out mechanism is unsolved rather than shortcut-solved,
and this is recorded as a negative result for the method.

**The design lesson is specific.** Current-centering was introduced to remove
the V1 (the instantaneous-geometry router) shortcut in which instantaneous geometry identified the mechanism, and
it worked: rung 1 falls to 0.0322. But because centering expresses every frame
as a signed displacement to the present, it *created* a rung-2 shortcut, where
one sufficiently old frame carries the whole answer. Fixing a leakage path at
one rung introduced a subtler one at the next, and only the ladder exposes it.

**Caveat on rung 3.** The hand-written rule never emits the `defer` option, so
its offline accuracy against targets that use deferral before onset is not
comparable to the learned rungs; its closed-loop score, 97.40% on V4's (the nominal-state revision) held-out
reverse, is the meaningful one. The shortcut verdict rests on rungs 1, 2, and
4, which share a training target convention.

Artifacts: `results/router/ladder/learned_recovery_v4.json`,
`peg_insertion.json`, and `peg_insertion_physical_only.json`.

### The ladder on REBOOT: the audit's negative control

The same four-rung audit was run on the external REBOOT benchmark: 2,072
real-robot trajectories across nine leave-one-object-out object families,
collected by another group. REBOOT prefixes are not current-centered, so its
final frame already carries signal and rung 1 sits at 0.5740 rather than at
zero; the matching rung-2 control is therefore the *endpoint pair*, the first
and last frame with no sequence encoder and no summary over the frames between.

| Rung | Control | macro-AUROC |
|---|---|---:|
| 1 | current frame | 0.5740 |
| 2 | endpoint pair | 0.6080 |
| -- | whole-prefix summary (mean, sd) | 0.7466 |
| 4 | unstructured GRU | 0.8113 |
| 4 | causal dynamics GRU | 0.8045 |

The endpoint pair recovers almost none of the recurrent models' advantage:
0.1966 macro-AUROC points behind, object-bootstrap 95% [0.1557, 0.2357]. This
benchmark's held-out object family is not identifiable from where a trajectory
started and ended. Together with PegInsertion this gives the audit two
negatives against one positive, so it does not merely fire everywhere; the
positive on LearnedRecovery-v4, where a memoryless model matches the recurrent
model exactly, is a property of that benchmark rather than of the audit.

One nuance is retained rather than smoothed over: the whole-prefix summary
reaches 0.7466, closing most of the 0.5740-to-0.8045 span with no recurrence.
The recurrent gain over it is real but small, 0.0579 [0.0031, 0.1382]. Most of
REBOOT's signal is in aggregate statistics rather than in dynamics.

**Defect found and fixed while running this.** `fit_one` never re-seeded before
constructing a model, so weight initialisation drew from the advancing global
RNG. Inserting the rung-2 control as the second method silently re-initialised
the three methods after it, moving the unstructured GRU from 0.8072 to 0.8291
and the causal GRU from 0.8353 to 0.8064 -- enough to flip the sign of their
comparison. Model comparisons must not depend on the order of the method list.
Seeding is now per (method, fold), and the causal-versus-unstructured
comparison returns to unresolved at -0.0067 [-0.0249, +0.0119], consistent with
the originally published +0.0282 [-0.0140, +0.0938]. The published REBOOT
figures were internally consistent; they were not robust to adding an arm.

Artifact: `results/a_plus_audit/reboot_ladder_v4_aggregate.json`.

### Matched rung set, and the verdict revision it forced

The ladder was first scored with a rung set that differed per benchmark: REBOOT
had an order-free summary control because its pipeline already provided one,
while the two simulated benchmarks did not. A ladder whose rungs differ per
benchmark cannot support a cross-benchmark claim, and in this case the omission
changed conclusions.

`MomentSummaryRouter` supplies the matched control. It reads every frame but has
no sequence encoder and no access to frame order, taking the mean and standard
deviation over the valid prefix. It is the strongest order-free control.

| Benchmark | r1 | r2 | r2b | r3 | r4 | best lower / r4 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| LearnedRecovery-v4 | 0.0322 | 1.0000 | 1.0000 | 0.1195 | 1.0000 | 1.000 | shortcut |
| PegInsertionSide-v1 | 0.0000 | 0.0909 | 0.0503 | 0.0000 | 0.4015 | 0.226 | none |
| REBOOT | 0.5740 | 0.6080 | 0.7466 | n/a | 0.8045 | 0.928 | shortcut |

Two revisions follow, and both are recorded rather than quietly absorbed.

REBOOT was previously reported here and in the README as a clean negative on
the strength of its endpoint-pair control at 0.6080. Its order-free summary
reaches 0.7466, or 0.928 of the recurrent score, which is above the 0.9
reporting line. REBOOT is a marginal positive, not a negative. The earlier
"one positive, two negatives" summary was wrong; the matched result is two
positives and one negative.

The LearnedRecovery-v4 verdict is strengthened rather than weakened: two
independent non-recurrent controls both reach 1.0000, so the shortcut is not an
artifact of how one control was built.

PegInsertion is the discriminating negative and becomes more decisive under the
matched set. Its order-free summary reaches 0.0503, weaker than its two-frame
control at 0.0909, and its strongest lower rung is 0.226 of the recurrent
score.

Two caveats attach to the interpretation. The 0.9 line is a reporting
convention chosen by this project, not a test with an error rate, and REBOOT at
0.928 sits close enough to it that the ratio should be quoted rather than the
verdict. And every REBOOT figure rests on three optimizer seeds whose
unstructured-GRU spread is 0.7754 to 0.8506, wider than several differences
under discussion; more seeds are required before those intervals carry weight.

### Replacing the ratio cut with a paired test, and ten-seed REBOOT

Two limitations of the previous ladder are resolved here and the verdict
history is recorded in full, because REBOOT has now moved twice.

**The criterion.** The audit previously called a lower rung "matching" when it
reached 0.9 of the recurrent score. That cut was chosen by this project and
carries no error rate. It is replaced by a paired bootstrap on the difference
between rung 4 and each lower rung, resampling the correlated unit: whole
episodes on the simulated benchmarks, object families on REBOOT. A lower rung
matches when that interval includes zero.

**Seeds.** REBOOT now uses ten optimizer seeds rather than three.

| Benchmark | Best lower | Rung 4 | rung4 - lower | Verdict |
|---|---:|---:|---|---|
| LearnedRecovery-v4 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | shortcut |
| PegInsertionSide-v1 | 0.0909 | 0.4015 | +0.3240 [+0.1344, +0.5231] | none |
| REBOOT (10 seeds) | 0.7482 | 0.8108 | +0.0626 [+0.0035, +0.1367] | none |

**REBOOT's verdict history.** It was first recorded as no shortcut, using an
endpoint-pair control and three seeds; that was unsound because the rung set
was not matched across benchmarks. It was then recorded as a shortcut, using
the matched order-free control at a ratio of 0.928 against the 0.9 cut; that
was unsound because the cut is arbitrary. It is now recorded as no shortcut
under the paired test at +0.0626 [+0.0035, +0.1367].

The criterion was changed to remove an acknowledged defect, not to obtain a
verdict, and the change was made before the ten-seed run completed. The verdict
moving as a consequence is disclosed rather than presented as the original
finding. The margin is thin -- a lower bound of +0.0035 and a ratio of 0.923 --
so both readings are reported and REBOOT should be described as a benchmark
where the order-free control comes close without matching. LearnedRecovery-v4
and PegInsertion are unambiguous under either criterion.

**Ten-seed REBOOT figures.** static 0.5734, endpoint pair 0.5995, order-free
summary 0.7482, unstructured GRU 0.8129, factorized GRU 0.8108. The factorized
and unstructured models are statistically indistinguishable at -0.0021
[-0.0123, +0.0069], tightening the three-seed interval of [-0.0249, +0.0119]
without changing its conclusion.

Artifacts: `results/router/ladder/*.json`,
`results/a_plus_audit/reboot_ladder_v5_aggregate.json`.

### The confusion pair does not replicate on PegInsertion

The permanent/temporary result was the one finding that survived every audit on
`LearnedRecovery-v4`: both non-recurrent arms failed the pair in opposite
directions while both recurrent arms solved both sides. Closed-loop on
PegInsertion, it does not hold.

| Arm | v4 permanent | v4 temporary | Peg permanent | Peg temporary |
|---|---:|---:|---:|---:|
| causal GRU | 0.9740 | 0.8438 | 0.4740 | 0.0052 |
| unstructured GRU | 0.9740 | 0.8420 | 0.5208 | 0.0156 |
| static, one frame | 0.0000 | 0.8438 | 0.5677 | 0.0312 |
| hand-written rule | 1.0000 | 0.0000 | 0.0000 | 0.0365 |

On permanent blockage the memoryless model is the *best* arm at 0.5677, ahead
of both recurrent models. Memory does not help there, which reverses the v4
result directly and matches what the offline per-condition accuracies already
suggested.

**A limitation of this experiment, stated plainly.** No PegInsertion recovery
specialists exist, so the nominal checkpoint was supplied for the nominal,
forward, and reverse roles. Every arm therefore shares identical and
recovery-incapable specialists. The relative comparison between arms remains
meaningful because they share them, but the absolute numbers are depressed and
the temporary column, where every arm scores at or below 0.0365, cannot support
a conclusion about routing: no arm had a policy capable of resuming after
clearance. The permanent column is the informative one.

The defensible reading is therefore narrow. Memory does not help on permanent
blockage in a contact-rich task, contradicting the v4 finding. The temporary
side is unresolved and needs Peg recovery specialists before it can be tested.

This is the third instance of one pattern: a result that appears fundamental on
`LearnedRecovery-v4` does not survive a harder or externally grounded
benchmark. The first was the held-out mechanism falling to a one-frame model,
the second was REBOOT's verdict depending on which control was nominated, and
this is the third.

### PegInsertion leaks blockage identity through episode timing

The static model on PegInsertion reaches 0.803 and 0.842 on permanent and
temporary blockage despite current-centering leaving its geometry input near
zero, and it beat both recurrent models closed-loop on permanent blockage. The
remaining input is `normalized_time`.

Zeroing that one column and retraining, with every other column, label, split
and group id unchanged:

| Model | Condition | With time | Without | Delta |
|---|---|---:|---:|---:|
| static MLP | permanent | 0.8034 | 0.6094 | -0.1940 |
| static MLP | temporary | 0.8424 | 0.6027 | -0.2398 |
| static MLP | positive ejection | 0.6033 | 0.6071 | +0.0038 |
| static MLP | negative ejection | 0.5989 | 0.6025 | +0.0036 |
| causal GRU | permanent | 0.7362 | 0.7468 | +0.0106 |
| causal GRU | temporary | 0.7731 | 0.7806 | +0.0075 |

The static model's entire blockage advantage is the clock. It loses 19 to 24
points on exactly the two blockage conditions and nothing at all on the two
ejection conditions, landing near the per-condition base rate. The recurrent
model is unaffected, so it was not relying on timing.

This also explains the closed-loop result. The memoryless arm won on permanent
blockage not by reading the physics but by reading episode duration, which
differs systematically between blockage and ejection episodes.

The lesson is transferable and concrete: a normalized-time feature in a
recovery benchmark leaks condition identity through episode duration, because
mechanisms that terminate the episode differently produce different clock
distributions. Any benchmark including such a feature should ablate it.

### Placement tolerance is loose enough that success can look like failure

Inspecting the temporary-blockage capture raised a question the artifacts could
not answer: the panel reports both goals placed, but the render reads as a
near-miss. `goals_completed` also latches, so a count of 2.0 does not by itself
prove the cubes are on their pads at that moment.

Measuring the physics directly at the resolution step and again at episode end:

| | red cube to goal | blue cube to goal |
|---|---:|---:|
| At resolution, step 76 | 0.0347 m | 0.0293 m |
| Threshold | 0.04 | 0.04 |
| At episode end, step 240 | 0.206 m | 0.219 m |

Three separate facts follow. The count is correct: both cubes are inside the
threshold at resolution, so the policy does complete both goals after the
obstruction clears, and this is genuine recovery rather than abandonment.

The margin is thin. At 0.035 and 0.029 against a 0.04 limit, with 5 cm cubes on
9 cm pads, a cube counted as placed can be overhanging the edge of its pad. The
success criterion admits placements that read as failures, which compounds the
task-difficulty limitation already recorded: a 4 cm tolerance with no
orientation requirement is generous, and the visual ambiguity is a symptom of
that rather than a rendering artifact.

The cubes are then displaced to roughly 0.21 m by the end of the episode. Those
164 steps are after scoring stops and are not measured, which is why captures
now hold the resolution frame. Before that change the figure showed cubes being
scattered after the episode had already been scored a success.

Captures now record `cube_to_goal_xy` at resolution and `cube_to_goal_xy_at_end`
so a latched count can always be checked against measured positions.
Artifact: `results/v4_place/router_temporary_block.json`.

**Render/annotation index mismatch, found and fixed.** `_render_frame` took
`image[0]` from the batched render while every annotation was read from
`args.capture_env_index`. Any capture taken from an environment other than 0
therefore produced a panel whose video and captions came from different
episodes. The temporary-blockage panel was captured from environment 2, whose
episode succeeds, while the video showed environment 0, whose episode never
resolves. The mismatch was visible as a robot that kept working after its
caption said the task was complete. Rendering now uses the capture index.
