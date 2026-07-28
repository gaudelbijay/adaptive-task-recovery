# atr.recovery

**Owns:** [docs/07-recovery-policy-design.md](../../../docs/07-recovery-policy-design.md)

## Scope

The recovery skill library (`regrasp`, `re-approach`, `step-recovery`,
`replan-and-retry`, `abort-to-safe-pose`) and the `Arbiter` that switches
between `task`, `recover`, and `abort`. v1 arbiter is a rule-based state
machine on the failure signal + confidence threshold; a learned
option-selection policy is a v2 stretch.

## Implements / consumes

- Implements `RecoverySkill` and `Arbiter` (`atr.interfaces`).
- Consumes `FailureSignal` from `atr.detection` and state from `atr.perception`.
- Produces `Action`s, sent to `atr.control`.

## Status

Not started — blocked on `atr.detection` producing a usable `FailureSignal`.
