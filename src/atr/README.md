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
| [`policies/q_learning.py`](policies/q_learning.py) | `train_q_table`, `learned_policy` — tabular Q-learning that discovers "attempt iff feasible" from reward, same `attempt_goal_fn`/`tray_slots` parameterization as `baselines.py` | `spikes/task_schema_draft/rl_policy.py` | D-041 |
| [`evaluation/harness.py`](evaluation/harness.py) | `compare_policies`/`bootstrap_ci` — the first real implementation of docs/10's "paired seeds, bootstrap CIs" statistical protocol. Env/policy-agnostic. | new (D-042) | D-042 |
| [`evaluation/splits.py`](evaluation/splits.py) | `InstructionSpec`, `TRAIN`/`HELD_OUT_PARAPHRASE`/`HELD_OUT_COMPOSITION`/`SPLITS` — the first queryable dataset-split registry, per docs/04's "hold out paraphrases and compositions". Pure data, zero simulator dependency. | strings copied verbatim from `test_instruction_parser.py` | D-044 |

This is D-013's original proposal (goal/constraint schema, oracle
feasibility, intent guard) plus the two schema questions that came up
during review and got resolved rather than deferred: `Goal.condition`
(D-026, kept as-is) and `Goal.depends_on` (D-037, was dead schema surface
until this promotion — see `goal_dependencies_satisfied()`'s docstring);
the language parser (D-038); zero-shot CLIP feasibility, calibrated not
general (D-039); env-agnostic policy-decision logic, unified from
four duplicated copies after that duplication caused a real,
now-fixed cross-variant bug (D-040); Q-learning (D-041), which
fixed an internal inconsistency D-040's own pattern exposed; a real
evaluation harness (D-042) implementing docs/10's paired-seed/bootstrap-CI
protocol for the first time — applied immediately to H2's original
static-vs-feasibility-aware comparison, which turned out to have zero
outcome variance across 30 seeds at this toy scale (reported honestly,
not hidden); and a queryable dataset-split registry (D-044), replacing
literal strings buried in test-function bodies with something any
evaluation code can enumerate programmatically.

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

DINOv2's self-supervised probe, the end-to-end pipeline, and every
environment variant — including each variant's own `attempt_goal()` (the
real, embodiment-specific low-level motion: Cartesian IK, joint-space
reach, or navigate-then-reach) and tray geometry, which
`policies/baselines.py`/`policies/q_learning.py` take as parameters
rather than owning themselves. None of those have made their own case
for promotion yet — each promotion so far (D-038 through D-044) was made
on that module's own evidence, not as a side effect of an earlier one,
and each carries whatever caveat its own evidence actually supports
(D-039's calibration-not-generalization note, D-040/D-041's "this
interface came from real implementations, not from docs/03's untested
pseudocode" — see each decision's own reasoning) — see
`spikes/task_schema_draft/README.md` for the full narrative.

[`configs/`](../../configs/) and [`data/`](../../data/) (added alongside
this package, D-032) are still empty — nothing here yet needs
configuration or a real dataset.
