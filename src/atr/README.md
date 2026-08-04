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
| [`envs/tidy_up_env.py`](envs/tidy_up_env.py) | `TidyUpEnv`/`TidyUpRegisteredEnv` — the canonical 5-object tabletop ManiSkill3 scene, registered as `TidyUp-v1` (was `TidyUpTaskSchemaDraft-v1`). | `spikes/task_schema_draft/tidy_up_env.py` | D-045 |
| [`envs/tidy_up_policies.py`](envs/tidy_up_policies.py) | `attempt_goal()` (real arm motion for the canonical env) + thin `static_policy`/`feasibility_aware_policy`/`naive_substitution_policy` wrappers over `policies/baselines.py`. Fixed a real duplication while promoting: tray/object positions are now derived from `tidy_up_env.py`'s `_OBJECT_SPECS`, not copy-pasted numbers. | `spikes/task_schema_draft/policy_baselines.py` | D-046 |
| [`envs/tidy_up_env_humanoid.py`](envs/tidy_up_env_humanoid.py) | Unitree G1 humanoid variant — same schema, joint-space reach instead of Cartesian IK. Registered as `TidyUp-Humanoid-v1`. | `spikes/task_schema_draft/tidy_up_env_humanoid.py` | D-047 |
| [`envs/tidy_up_humanoid_policies.py`](envs/tidy_up_humanoid_policies.py) | Same policy API as `tidy_up_policies.py`, for the humanoid env. `_TRAY_POSITION`'s z left as-is (0.698, not 0.755) — checked first: it's a real settled-vs-spawn-height difference, not a stale duplicate. | `spikes/task_schema_draft/policy_baselines_humanoid.py` | D-047 |
| [`envs/tidy_up_env_replicacad.py`](envs/tidy_up_env_replicacad.py) | Real ManiSkill3 `ReplicaCADSetTableTrain` apartment, mobile Fetch robot, real YCB objects. Registered as `TidyUp-ReplicaCAD-v1`. | `spikes/task_schema_draft/tidy_up_env_replicacad.py` | D-048 |
| [`envs/navigation.py`](envs/navigation.py) | Generic grid + Dijkstra path planner — no project-internal dependency. | `spikes/task_schema_draft/navigation.py` | D-048 |
| [`envs/tidy_up_replicacad_policies.py`](envs/tidy_up_replicacad_policies.py) | Same policy API, navigating (not just reaching) to each goal. `_TRAY_POSITION`/`_TRAY_HALF_SIZES` were already imported, not duplicated, before promotion — no fix needed there. | `spikes/task_schema_draft/policy_baselines_replicacad.py` | D-048 |
| [`envs/tidy_up_env_replicacad_humanoid.py`](envs/tidy_up_env_replicacad_humanoid.py) | G1 fixed-base, placed (not navigating) in the same real apartment. Registered as `TidyUp-ReplicaCAD-Humanoid-v1`. Closes out all four embodiment/scene variants. | `spikes/task_schema_draft/tidy_up_env_replicacad_humanoid.py` | D-049 |
| [`envs/tidy_up_replicacad_humanoid_policies.py`](envs/tidy_up_replicacad_humanoid_policies.py) | Same policy API, arm-reach only (no navigation). Positions already imported, not duplicated. | `spikes/task_schema_draft/policy_baselines_replicacad_humanoid.py` | D-049 |
| [`pipeline.py`](pipeline.py) | `run_end_to_end_episode()` — language parsing, real vision-based feasibility, and a learned policy combined into one real episode. Last of the six build-up stages promoted. Uses `atr.policies.q_learning.greedy_action()` (new, D-050) instead of a duplicated argmax lookup. | `spikes/task_schema_draft/end_to_end.py` | D-050 |
| [`control/ik_solver.py`](control/ik_solver.py) | `solve_right_arm_ik()`/`best_reachable_distance()` — real analytic-Jacobian IK on `pinocchio`, deterministic, verified against ManiSkill's own kinematics. Zero project-internal dependency, plain `git mv`. | `spikes/task_schema_draft/ik_solver.py` | D-051 |
| [`envs/capture_episode_subprocess.py`](envs/capture_episode_subprocess.py) | Standalone script (never imported, run via subprocess) that captures one render-producing reset in its own fresh process — works around D-022's confirmed upstream ManiSkill3 rendering bug. Promoted even though its main caller (`dinov2_probe.py`) isn't ready, same situation D-039 handled for `device_utils.py`. | `spikes/task_schema_draft/capture_episode_subprocess.py` | D-052 |

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
not hidden); a queryable dataset-split registry (D-044), replacing
literal strings buried in test-function bodies with something any
evaluation code can enumerate programmatically; the canonical task
environment itself (D-045), the first genuinely simulator-specific
architecture promoted here now that D-033 has formally selected
ManiSkill3 — its registered id dropped the "draft" qualifier
(`TidyUpTaskSchemaDraft-v1` → `TidyUp-v1`) at promotion time; that
env's own policy-facing API (D-046), which fixed a real duplication
along the way — tray/object positions are now derived from the env's own
`_OBJECT_SPECS`, not separately copy-pasted numbers that could silently
drift; the Unitree G1 humanoid variant (D-047), where a similar-looking
position mismatch turned out to be a real, legitimate difference (settled
vs. spawn height) rather than the same kind of bug — checked, not assumed,
before deciding to leave it alone; the ReplicaCAD + Fetch variant
(D-048), with real navigation (`navigation.py`, generic, promoted
alongside) — this one had no duplication bug to find at all, since it
has no `_OBJECT_SPECS`-equivalent to duplicate from in the first place;
the fourth and final embodiment/scene variant, G1 fixed-base in the
same real apartment (D-049) — same clean pattern as D-048, nothing to
fix; and the end-to-end pipeline itself (D-050), the last of the six
build-up stages, once everything it depended on already was promoted —
also fixed a small duplicated argmax lookup along the way
(`greedy_action()`, shared with `policies/q_learning.py`'s
`learned_policy()`); the real analytic-Jacobian IK solver (D-051,
`control/`) — already zero-dependency, plain `git mv`; and the
subprocess capture script (D-052), promoted despite its main caller
(`dinov2_probe.py`) not being ready, since it independently serves the
already-promoted CLIP kitchen_sink tests too.

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

Only `dinov2_probe.py` — the one module flagged early on as not ready
(D-039's write-up: weaker evidence than CLIP, one scene layout, never
wired into a live decision loop) and still true. Everything else that
started in `spikes/task_schema_draft/` has now either been promoted or
explicitly evaluated and held back. Each promotion so far (D-038 through D-052)
was made on that module's own evidence, not as a side effect of an
earlier one, and each carries whatever caveat its own evidence actually
supports (D-039's calibration-not-generalization note, D-040/D-041's
"this interface came from real implementations, not from docs/03's
untested pseudocode" — see each decision's own reasoning) — see
`spikes/task_schema_draft/README.md` for the full narrative.

[`configs/`](../../configs/) and [`data/`](../../data/) (added alongside
this package, D-032) are still empty — nothing here yet needs
configuration or a real dataset.
