# `src/atr/`

Committed project architecture. As of 2026-08-02 (D-037), no longer empty:
D-013's core schema has been reviewed and promoted here.

## What's here

| Module | Contents | Promoted from | Decision |
|---|---|---|---|
| [`language/goal_graph.py`](language/goal_graph.py) | `Goal`, `Constraint`, `GoalGraph` dataclasses, `canonical_example()`, `dependent_goals_example()` | `spikes/task_schema_draft/goal_graph.py` | D-037 |
| [`feasibility/oracle.py`](feasibility/oracle.py) | `goal_feasible`, `goal_achieved`, `goal_dependencies_satisfied`, `constraint_violated`, `evaluate_goal_graph` | `spikes/task_schema_draft/oracle_feasibility.py` | D-037 |
| [`constraints/intent_guard.py`](constraints/intent_guard.py) | `validate_action` | `spikes/task_schema_draft/intent_guard.py` | D-037 |
| [`language/instruction_parser.py`](language/instruction_parser.py) | `parse_instruction()` — controlled-grammar text → `GoalGraph` | `spikes/task_schema_draft/instruction_parser.py` | D-038 |
| [`feasibility/clip_feasibility.py`](feasibility/clip_feasibility.py) | `visual_object_exists()` — zero-shot CLIP feasibility from a rendered frame. **Calibrated per object/scene, not generalizing** — read the module docstring before trusting it as general. | `spikes/task_schema_draft/clip_feasibility.py` | D-039 |
| [`device_utils.py`](device_utils.py) | `resolve_torch_device()` — CUDA-with-CPU-fallback for torch models | `spikes/task_schema_draft/device_utils.py` | D-039 |
| [`policies/baselines.py`](policies/baselines.py) | `static_policy`, `feasibility_aware_policy`, `naive_substitution_policy` — env-agnostic policy-decision logic, parameterized by an `attempt_goal_fn` each spike env supplies | unified from 4 near-identical `spikes/task_schema_draft/policy_baselines*.py` copies | D-040 |

This is D-013's original proposal (goal/constraint schema, oracle
feasibility, intent guard) plus the two schema questions that came up
during review and got resolved rather than deferred: `Goal.condition`
(D-026, kept as-is) and `Goal.depends_on` (D-037, was dead schema surface
until this promotion — see `goal_dependencies_satisfied()`'s docstring);
the language parser (D-038); zero-shot CLIP feasibility, calibrated not
general (D-039); and env-agnostic policy-decision logic, unified from
four duplicated copies after that duplication caused a real,
now-fixed cross-variant bug (D-040).

## Review status — read before trusting this as "reviewed"

**Self-resolved by the project owner (D-037), not independently reviewed
by the teammate this schema was actually written for.** See
[`ai-notes/review-request-task-schema.md`](../../ai-notes/review-request-task-schema.md)
for the full resolution of each open question, and its status banner for
what "self-resolved" does and doesn't mean here. Toy-scale evidence
throughout — promotion changed where this code lives and its accept
status, not the underlying evidence's scale. See
`ai-notes/decisions.md` D-013–D-037 for the full history.

## What's still in `spikes/task_schema_draft/`, not here

DINOv2's self-supervised probe, the tabular Q-learning algorithm
(`rl_policy.py`), the end-to-end pipeline, and every environment variant
— including each variant's own `attempt_goal()` (the real,
embodiment-specific low-level motion: Cartesian IK, joint-space reach, or
navigate-then-reach) and tray geometry, which `policies/baselines.py`
takes as parameters rather than owning itself. None of those have made
their own case for promotion yet — each promotion so far (D-038, D-039,
D-040) was made on that module's own evidence, not as a side effect of
an earlier one, and each carries whatever caveat its own evidence
actually supports (D-039's calibration-not-generalization note, D-040's
"this interface came from four real implementations, not from
docs/03's untested pseudocode" — see that decision's own reasoning) —
see `spikes/task_schema_draft/README.md` for the full narrative.

[`configs/`](../../configs/) and [`data/`](../../data/) (added alongside
this package, D-032) are still empty — nothing here yet needs
configuration or a real dataset.
