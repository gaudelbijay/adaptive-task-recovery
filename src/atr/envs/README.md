# atr.envs

**Owns:** [docs/04-simulation-environment-maniskill.md](../../../docs/04-simulation-environment-maniskill.md)

## Scope

ManiSkill3 task environments (`PushRecoveryStand`, `PickPlaceRecovery`,
`DoorOpenRecovery`, `CarryWalkRecovery`) and the `FailureInjector` that applies
deterministic, seeded failures inside `env.step()`.

## Implements / consumes

- Implements `RobotInterface` (`atr.interfaces`) over the ManiSkill3 `BaseEnv`/`BaseAgent` API.
- Produces `Observation`s and `FailureEvent`s consumed by `atr.perception` and `atr.detection`.
- Depends on nothing else in `src/atr/` — this is the base layer everything else builds on.

## Status

Not started. First concrete step (per [STATUS.md](../../../STATUS.md)): install ManiSkill3,
import a humanoid asset, pass the standing-stability smoke test, then build
`PushRecoveryStand`.
