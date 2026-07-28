# Media status

The existing `architecture-diagram.drawio` and rendered preview files describe
the superseded humanoid failure-detection and recovery architecture. They are
retained temporarily as design history but are not authoritative and are no
longer linked from the project README.

The replacement diagram should match [the current system architecture](../docs/03-system-architecture.md):
self-supervised visual encoder, language goal graph, world-change belief,
per-goal feasibility estimator, adaptive policy, intent guard, and simulated
humanoid embodiment/skill interface.
