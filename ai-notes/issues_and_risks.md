# Issues and Risks

Last updated: 2026-07-24

## Active

| ID | Type | Severity | Description | Mitigation / next check |
|---|---|---|---|---|
| R-001 | Risk | High | A suitable humanoid asset may be difficult to import or stabilize in ManiSkill. | Validate one asset during Phase 0; keep a simpler supported robot as a pipeline fallback. |
| R-002 | Risk | Medium | Dependency or GPU support may differ across development machines. | Pin versions and document a tested setup after the first successful example. |
| R-003 | Risk | High | The project scope may grow before a baseline works. | Hold Phase 1 to one task and one baseline until its exit criteria pass. |
| R-004 | Risk | Medium | Injected failures may be too artificial to support useful generalization claims. | Hold out failure types and evaluate naturally occurring policy failures separately. |
| I-001 | Open question | Medium | The Python, PyTorch, ManiSkill, and SAPIEN version combination is not selected. | Resolve during initial setup and record it in project configuration. |
| I-002 | Open question | Medium | The first humanoid model is not selected. | Compare asset availability, licensing, import effort, and standing stability. |

## Resolved

| ID | Resolution date | Resolution |
|---|---|---|
| I-000 | 2026-07-24 | Project scope set to simulation-only. |

## Update rules

- Give each item a stable ID.
- Add a concrete mitigation or next check.
- Move resolved items to the resolved table; do not silently delete them.
- Promote a risk into `todo.md` when it has a concrete action that should happen
  in the current phase.
