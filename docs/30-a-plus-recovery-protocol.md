# A/A+ recovery protocol (frozen before development)

This protocol replaces the previous goal of improving the V28 aggregate.  V28
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
4. Restricted-RGB V19 remains context evidence and is never presented as an
   input-matched primary baseline for a state-observation router.

## Stopping rule

The README headline changes only if every frozen criterion passes on the
untouched confirmation bank.  Otherwise the repository records the candidate
as rejected, including the failed condition and uncertainty interval.  No
subset, seed, or favorable OOD profile can substitute for the frozen gate.

## V2 temporal-composition addendum

The V1 audit found that the five simulator mechanisms are almost perfectly
identified from one task-relative state: the causal and unstructured GRUs
produced the same 878/960 safe successes and the static MLP was one episode
behind. V1 remains rejected. Further tuning on that representation cannot
support a causal-memory claim.

Before V2 training or evaluation, the follow-up protocol is frozen in
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

V2 is rejected before closed-loop evaluation. Across three optimizer seeds,
the causal factorized router reached 100% held-out reverse accuracy and the
unstructured GRU reached 0%, but the static factorized MLP reached 66.11%,
above the frozen 40% shortcut ceiling. Inspection of the mechanically named
feature contract found that V2 centered the first 42 actor/mechanism geometry
dimensions but omitted dimensions 42:57, which encode TCP position relative
to cubes, goals, and the protected object.

V3 is frozen in `configs/a_plus_recovery_gate_v3_full_geometry.json` before
training. It centers the complete 57-dimensional geometry prefix and changes
no numerical threshold. Its selection family is `328000000`; its untouched
confirmation family is `332000000`. V2 remains a machine-recorded failed
candidate and is not pooled with V3.

### V3 confirmation rejection and V4 nominal-controller correction

V3 was opened once on its untouched `332000000` confirmation family. The
causal router achieved 2655/2880 (92.19%) safe successes with 51/2880 (1.77%)
violations, versus 2369/2880 (82.26%) for the strongest non-oracle baseline.
The 9.93-point gain had a 95% Newcombe interval of [7.54, 12.29] points. It
also achieved 573/576 safe successes on the held-out reverse condition.
Nevertheless, nominal safe success was only 456/576 (79.17%), below the frozen
82% worst-condition floor. V3 is therefore rejected and is never rerun as an
untouched result.

V4 is preregistered in `configs/a_plus_recovery_gate_v4_nominal_state.json`.
It changes only the controller shared by nominal execution and temporary
recovery after clearance: the V19 RGB controller is replaced with a dedicated
LearnedRecovery-v4 state PPO trained on nominal episodes. The exact selected
checkpoint is shared by every router baseline. The 57-dimensional temporal
representation, specialists, router calibration, 36-step safe hold, OOD axes,
and every numerical threshold remain unchanged. V4 uses `329000000` for
selection and reserves `333000000` for a once-only untouched confirmation.

### V4 selection rejection and V5 controller selection

V4 is rejected before confirmation. Two of its three state PPO seeds learned
0% nominal success; the best seed's frozen checkpoint achieved only 100/192
(52.08%) safe successes on the `329000000` nominal selection episodes. There
were zero violations, but the 82% condition floor failed decisively. The
reserved `333000000` family remains untouched.

The same selection family was used for a declared five-way screen of the
existing shared nominal controller. V19 seed 9351 achieved 166/192 (86.46%)
safe successes, compared with 135/192 and 152/192 for the other individual
seeds, 150/192 for the mean ensemble, and 145/192 for the median ensemble. V5
freezes seed 9351's exact checkpoint and hash in
`configs/a_plus_recovery_gate_v5_selected_nominal.json`. It changes no router,
specialist, representation, hold, or numerical threshold, and uses fresh
`330000000` selection and untouched `334000000` confirmation families.

### V5 confirmation rejection and V6 option-specific library

V5 was opened once on `334000000`. It achieved 2623/2880 (91.08%) safe
successes and beat the matched unstructured model by 7.57 points with a 95%
Newcombe interval of [5.16, 9.96] points. It is nevertheless rejected: 87/2880
violations is 3.0208%, one episode beyond the 3.00% cap, and temporary recovery
was 468/576 (81.25%), five episodes below the 82% condition floor. These are
not rounded into passes, and no V5 OOD confirmation is run.

V6 freezes an option-specific shared controller library in
`configs/a_plus_recovery_gate_v6_option_specific_nominal.json`. Nominal option
0 keeps V5's selected seed-9351 controller; temporary post-clearance option 4
uses the seed-1788 controller that was stronger on that recovery role. The
library is identical for every router. V6 changes no routing model, specialist,
feature, calibration, hold, or numerical criterion. It uses `335000000` for
selection and reserves `339000000` for once-only confirmation.

### V6 selection rejection and V7 evidence-conditioned defer

V6 is rejected before confirmation. It passed overall safe success, violation,
gain, uncertainty, held-out reverse, and temporary recovery, but nominal safe
success fell to 435/576 (75.52%) on the fresh `335000000` selection family.
The reserved `339000000` family remains untouched.

Development on the opened selection family showed that the universal 36-step
defer, not recovery routing, was placing the nominal controller outside its
training-state distribution. V7 retains 36 as a maximum hold but releases it
when the method's own calibrated posterior confirms nominal execution. Later
events remain revisable. Every method receives the same release rule; no
mechanism label is consulted. A declared controller screen chose V19 seed 1788,
which achieved 167/192 (86.98%) nominal safe successes and 893/960 (93.02%)
across all conditions in the causal development run. V7 is frozen for fresh
`336000000` selection and untouched `340000000` confirmation in
`configs/a_plus_recovery_gate_v7_evidence_release.json`.
