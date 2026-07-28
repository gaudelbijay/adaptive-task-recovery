# Issues and Risks

Last updated: 2026-07-26

## Active

| ID | Type | Severity | Description | Mitigation / next check |
|---|---|---|---|---|
| R-005 | Risk | High | “Feasibility” may collapse into detecting intervention labels rather than estimating reachability. | Use reversible/neutral controls, counterfactual pairs, and oracle-regret evaluation. |
| R-006 | Risk | High | “Original intent” is underspecified in free-form language. | Begin with a formal goal graph and controlled language; bound claims explicitly. |
| R-007 | Risk | High | Privileged simulator state or template artifacts may leak feasibility labels. | Isolate label channels and audit seeds, pixels, timing, and language tokens. |
| R-008 | Risk | Medium | RL variance and large visual encoders may exceed available compute. | Validate with oracle state and frozen small encoders before scaling. |
| R-009 | Risk | Medium | An intervention may be called irreversible only because the planner times out. | Separate `unknown` from `infeasible`; validate oracle cases and bounds. |
| R-010 | Risk | Medium | The intent guard may trivially avoid violations by doing nothing. | Report feasible-goal completion and selective coverage alongside violations. |
| R-011 | Risk | High | Humanoid controller failures may be confused with high-level goal infeasibility. | Use a skill interface, repeated/oracle reachability labels, and separate error decomposition. |
| R-012 | Risk | Medium | Humanoid simulation and visual RL may exceed the compute budget. | Prototype logic cheaply, reuse low-level skills, freeze encoders initially, and retain humanoid as the final gate. |
| I-003 | Open question | High | Primary humanoid-capable simulator and asset are not selected. | Run a Phase 0 spike against documented selection criteria. |
| I-004 | Open question | Medium | Language backbone and SSL visual baselines are not selected. | Choose only after task schema and compute budget are known. |

## Resolved or superseded

| ID | Resolution date | Resolution |
|---|---|---|
| R-001 | 2026-07-26 | Original asset-import risk superseded by R-011/R-012 and the new skill-interface design. |
| R-004 | 2026-07-26 | Replaced old injected-failure concern with intervention validity and leakage risks. |
| I-002 | 2026-07-26 | Original model-selection issue superseded by the broader humanoid simulator/asset decision in I-003. |
| I-000 | 2026-07-24 | Project scope set to simulation-only. |
