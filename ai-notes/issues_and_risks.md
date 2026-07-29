# Issues and Risks

Last updated: 2026-07-29

## Active

| ID | Type | Severity | Description | Mitigation / next check |
|---|---|---|---|---|
| R-005 | Risk | High | “Feasibility” may collapse into detecting intervention labels rather than estimating reachability. Concretely present in the current draft: `goal_feasible()` in `spikes/task_schema_draft/oracle_feasibility.py` is a direct object-existence query — functionally almost identical to detecting the intervention label itself, not an estimate of reachability. Acceptable for now only because it's a privileged-state oracle used to validate plumbing (D-014), not a learned feasibility model. | Use reversible/neutral controls, counterfactual pairs, and oracle-regret evaluation. When a learned feasibility estimator is built, it must not have direct access to the "did an intervention fire" signal, or this risk reproduces exactly. |
| R-006 | Risk | High | “Original intent” is underspecified in free-form language. | Begin with a formal goal graph and controlled language; bound claims explicitly. |
| R-007 | Risk | High | Privileged simulator state or template artifacts may leak feasibility labels. | Isolate label channels and audit seeds, pixels, timing, and language tokens. |
| R-008 | Risk | Medium | RL variance and large visual encoders may exceed available compute. | Validate with oracle state and frozen small encoders before scaling. |
| R-009 | Risk | Medium | An intervention may be called irreversible only because the planner times out. | Separate `unknown` from `infeasible`; validate oracle cases and bounds. |
| R-010 | Risk | Medium | The intent guard may trivially avoid violations by doing nothing. | Report feasible-goal completion and selective coverage alongside violations. |
| R-011 | Risk | High | Humanoid controller failures may be confused with high-level goal infeasibility. Concretely observed in the ManiSkill3 spike (2026-07-28): a naive constant-hold action falls within ~0.5s even with zero injected disturbance — a controller-quality problem that would look identical to "infeasible" without careful separation. | Use a skill interface, repeated/oracle reachability labels, and separate error decomposition. |
| R-012 | Risk | Medium | Humanoid simulation and visual RL may exceed the compute budget. Partially confirmed: no CUDA on the primary dev machine, so GPU-vectorized parallel sim isn't available there — CPU sim is workable for single-env dev only. | Prototype logic cheaply, reuse low-level skills, freeze encoders initially, retain humanoid as the final gate, and budget for a CUDA machine/cloud GPU before any parallel RL training phase. |
| I-003 | Open question | Low | Primary humanoid-capable simulator and asset are not formally selected. ManiSkill3 spiked (2026-07-28, `spikes/maniskill_humanoid_spike/README.md`, D-009/D-010/D-011): humanoid support, seeding, privileged state, object-level interventions, RGB-D observations, and basic reach/grasp all confirmed working. Known gap: `mplib`-based canned motion planning doesn't build on Apple Silicon macOS (worked around via `pinocchio`-based IK). Only Isaac Lab comparison remains untested. | Spike Isaac Lab against the same criteria for a real head-to-head, or decide the ManiSkill3 evidence is sufficient and formally select it (D-006 requires the spike step, not necessarily a second candidate). |
| I-004 | Open question | Medium | Language backbone and SSL visual baselines are not selected. | Choose only after task schema and compute budget are known. |

## Resolved or superseded

| ID | Resolution date | Resolution |
|---|---|---|
| R-001 | 2026-07-26 | Original asset-import risk superseded by R-011/R-012 and the new skill-interface design. |
| R-004 | 2026-07-26 | Replaced old injected-failure concern with intervention validity and leakage risks. |
| I-002 | 2026-07-26 | Original model-selection issue superseded by the broader humanoid simulator/asset decision in I-003. |
| I-000 | 2026-07-24 | Project scope set to simulation-only. |
