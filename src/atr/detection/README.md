# atr.detection

**Owns:** [docs/06-failure-taxonomy-and-detection.md](../../../docs/06-failure-taxonomy-and-detection.md)

## Scope

`FailureMonitor` implementations, built in this order (don't skip ahead — each
is the baseline the next must beat): threshold baseline → ensemble
dynamics-disagreement → sequence anomaly model → multi-modal fusion.

## Implements / consumes

- Implements `FailureMonitor` (`atr.interfaces`).
- Consumes the state vector from `atr.perception` (an `ObservationWindow`).
- Produces `FailureSignal`, consumed by `atr.recovery`'s `Arbiter`.

## Status

Not started — blocked on `atr.envs` (need injected failures with ground-truth
labels) and `atr.perception` (need a state vector to consume).
