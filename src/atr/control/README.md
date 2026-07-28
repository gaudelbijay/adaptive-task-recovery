# atr.control

**Owns:** whole-body controller / action-interface section of [docs/03-system-architecture.md](../../../docs/03-system-architecture.md) §2.

## Scope

Maps desired task-space/joint targets from `atr.recovery` (and the nominal
task policy) down to `ManiSkill3` `agent.set_action()` calls, respecting
balance and contact constraints. Likely wraps a whole-body QP controller for
the step-recovery skill specifically — see [docs/02-background-and-related-work.md](../../../docs/02-background-and-related-work.md) §3.

## Implements / consumes

- Consumes `Action` (task-space/joint targets) from `atr.recovery` and the task policy.
- Calls `RobotInterface.apply_action()` (`atr.interfaces`), implemented by `atr.envs`.

## Status

Not started.
