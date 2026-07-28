"""A minimal, deterministic scripted physical intervention: one external push.

Part of the ManiSkill3 simulator-selection spike (see ../README.md and
docs/04-benchmark-environment.md "Selection requirements" / D-006 in
ai-notes/decisions.md). This is deliberately NOT an implementation of the
project's `WorldIntervention` Protocol (docs/04 §"Intervention API") — it
only tests one narrow thing: can this simulator apply a scripted physical
event to a humanoid at a reproducible, seeded moment in an episode? The
actual object-level interventions (object removed, container broken, route
blocked, ...) are a separate, later validation once a simulator is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PushInterventionEvent:
    """Ground-truth record of the applied push, for spike-eval logging only."""

    onset_step: int
    severity: float
    force: np.ndarray = field(repr=False)


class ScriptedPushIntervention:
    """Applies one external-force push per episode, at a random control step.

    One instance per episode. Construct with a `np.random.Generator` derived
    from the environment's episode seed so the push is exactly reproducible
    given that seed (see `HumanoidStandSpikeEnv._initialize_episode`).
    """

    def __init__(
        self,
        rng: np.random.Generator,
        onset_step_range: tuple[int, int] = (20, 80),
        force_magnitude_range: tuple[float, float] = (80.0, 400.0),
        severity: float | None = None,
    ):
        self.onset_step = int(rng.integers(*onset_step_range))
        self.severity = float(rng.uniform(0.0, 1.0)) if severity is None else float(severity)
        lo, hi = force_magnitude_range
        magnitude = lo + self.severity * (hi - lo)
        angle = float(rng.uniform(0.0, 2 * np.pi))
        self.force = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64) * magnitude
        self._applied = False

    def maybe_trigger(self, step_idx: int) -> PushInterventionEvent | None:
        """Call once per control step. Returns an event the one time the push fires."""
        if not self._applied and step_idx >= self.onset_step:
            self._applied = True
            return PushInterventionEvent(
                onset_step=self.onset_step, severity=self.severity, force=self.force.copy()
            )
        return None
