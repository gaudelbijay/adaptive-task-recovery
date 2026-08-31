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
