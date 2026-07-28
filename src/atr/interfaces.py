"""Cross-module contracts. See docs/03-system-architecture.md §3.

Every module (envs, perception, detection, recovery, control) is built and
tested against these Protocols, not against each other's internals. Changing
a Protocol here is a cross-module design decision — update
docs/03-system-architecture.md and the affected module READMEs in the same
change.
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol


class Observation(Protocol):
    """Placeholder — replace with the real observation type once envs/ exists."""


class Action(Protocol):
    """Placeholder — replace with the real action type once control/ exists."""


class ContactState(Protocol):
    """Placeholder — per-link contact flags/forces."""


class ObservationWindow(Protocol):
    """A sliding window of Observations, as consumed by FailureMonitor."""


class FailureSignal(Protocol):
    is_failure: bool
    failure_type: Optional[str]
    confidence: float
    latency_ms: float


class RobotInterface(Protocol):
    def get_observation(self) -> Observation: ...
    def apply_action(self, action: Action) -> None: ...
    def get_contact_state(self) -> ContactState: ...
    def emergency_stop(self) -> None: ...


class FailureMonitor(Protocol):
    def update(self, obs_window: ObservationWindow) -> FailureSignal: ...


class RecoverySkill(Protocol):
    def can_initiate(self, obs: Observation, failure_signal: FailureSignal) -> bool: ...
    def step(self, obs: Observation) -> Action: ...
    def is_terminated(self, obs: Observation) -> bool: ...
    def succeeded(self, obs: Observation) -> bool: ...


class Arbiter(Protocol):
    def select(
        self, obs: Observation, failure_signal: FailureSignal
    ) -> Literal["task", "recover", "abort"]: ...
