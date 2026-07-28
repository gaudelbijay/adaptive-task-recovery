# atr.perception

**Owns:** perception section of [docs/03-system-architecture.md](../../../docs/03-system-architecture.md) §2; decision record: [ai-notes/decisions.md](../../../ai-notes/decisions.md) D-004.

## Scope

Turns raw `atr.envs` observations into the state vector `s_t` (proprioceptive +
task-relevant object/goal features) consumed by `atr.detection` and
`atr.recovery`.

**v1:** privileged simulator state only (object pose, contact, joint state) — no
learned perception. **Later, additive:** frozen pretrained visual backbone
(e.g. DINOv2) as an extra feature source. Out of scope, by decision: training a
custom self-supervised visual encoder, or using a vision-language model — see D-004
for why.

## Implements / consumes

- Consumes `Observation` from `atr.envs`.
- Produces the state vector consumed by `FailureMonitor.update()` (`atr.detection`) and recovery skills (`atr.recovery`).

## Status

Not started — blocked on `atr.envs` producing real observations.
