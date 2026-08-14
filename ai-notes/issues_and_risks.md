# Issues and Risks

Last updated: 2026-08-13 (D-114)

## Active

| ID | Type | Severity | Description | Mitigation / next check |
|---|---|---|---|---|
| R-005 | Risk | High | “Feasibility” may collapse into detecting intervention labels rather than estimating reachability. Concretely present in the current draft: `goal_feasible()` in `src/atr/feasibility/oracle.py` (promoted from `spikes/task_schema_draft/oracle_feasibility.py`, D-037) is a direct object-existence query — functionally almost identical to detecting the intervention label itself, not an estimate of reachability. Acceptable for now only because it's a privileged-state oracle used to validate plumbing (D-014), not a learned feasibility model. Still true after D-020/D-023: those give a real (imperfect) *perceptual* feasibility signal instead of privileged state, which is the actual mitigation this risk calls for, but only for the two objects/scenes calibrated so far. | Use reversible/neutral controls, counterfactual pairs, and oracle-regret evaluation. When a learned feasibility estimator is built, it must not have direct access to the "did an intervention fire" signal, or this risk reproduces exactly. |
| R-006 | Risk | High | “Original intent” is underspecified in free-form language. | Begin with a formal goal graph and controlled language; bound claims explicitly. |
| R-007 | Risk | High | Privileged simulator state or template artifacts may leak feasibility labels. | Isolate label channels and audit seeds, pixels, timing, and language tokens. |
| R-008 | Risk | Medium | RL variance and large visual encoders may exceed available compute. Partially addressed for the *decision*-level RL policy (D-025 — tabular Q-learning, ~19s on CPU), which sidesteps this by operating on privileged state rather than pixels; a pixel-conditioned policy would still face this risk in full. | Validate with oracle state and frozen small encoders before scaling. |
| R-009 | Risk | Medium | An intervention may be called irreversible only because the planner times out. | Separate `unknown` from `infeasible`; validate oracle cases and bounds. |
| R-010 | Risk | Low | The intent guard may trivially avoid violations by doing nothing. First toy test (D-015, `src/atr/constraints/intent_guard.py`, promoted D-037) only exercised the *easy* case — blocking an action that never earned goal credit anyway. **Real tension tested 2026-08-04 (D-058):** built the two constructible scenarios this mitigation note asked for. (1) A goal in direct target conflict with a matching `never_move` constraint — confirmed the guard does NOT over-block it (the goal wins), the literal concern this risk describes. (2) A genuinely opposite finding along the way: *without* privileged state, the guard was too *permissive* — a conditional goal (`Goal.condition`, D-026) exempted its target object from a constraint even while its condition didn't hold, since "is this a goal target" only checked declaration, not current feasibility. Fixed by threading `state` through `validate_action()`/`goal_feasible()`; `naive_substitution_policy` now passes it. Downgraded from Medium since both constructible scenarios are now tested and the found gap is fixed. **The physical-obstruction gap closed 2026-08-09 (D-082–D-087):** quantified the guard's aggregate recall/violation-rate trade-off, extended `validate_action()` to check predicted side effects (not just the named target), built a real swept-corridor effect predictor, and wired it into the real Fetch navigation stack — reaching for a legitimate mug while incidentally passing near a protected glass is representable and tested end-to-end. **The execution-contract remainder closed 2026-08-12/13 (D-091–D-100, see update notes below):** `_navigate_to()` now screens the real planned route before driving, stops safely (zero motion) if it's rejected, and — since D-092 — searches a constraint-aware detour around the predicted hazard before falling back to stopping. D-096–D-100 validated this with fully live Fetch execution (no mocks) across a stop-vs-replan safety-matched-recall comparison (D-097), three hazard locations on the original route (D-098), the second goal's route (D-099), and a second protected-object type to confirm the behavior follows the `GoalGraph` constraint rather than a hardcoded name (D-100) — every case completed the legitimate goal with exactly `0.0 m` protected-object displacement. Still approximates objects as spheres/points, not full robot-link collision geometry, and every live validation so far is one scene/layout/seed, not a distribution. **Broadened beyond single-scenario validation 2026-08-13 (D-108–D-114, see update notes below):** D-108 ran the first real multi-seed benchmark of the whole reachable-target stack (30 paired seeds, `static` vs `oracle_feasibility`) and found a real, paired-bootstrap-significant reduction in wasted steps with identical recall. D-109–D-114 did the same for the unreachable/no-route branch (D-107): live real-apartment execution across 3 seeds, 2 object identities, and 2 disconnected occupancy regions, all producing the same honest zero-motion fail-stop. Distinct scene *layouts* remain untested — `TidyUp-ReplicaCAD-v1` pins one `build_config_idx` and R-014 blocks trusting a new one without dedicated validation work. | Report feasible-goal completion and selective coverage alongside violations (done via D-082's aggregate metrics, D-094's `navigation_replans`/`navigation_safety_blocks`, D-110's `navigation_failures`). Remaining: a second, R-014-cleared scene layout; extend swept-corridor geometry beyond spheres/points if a scenario needs it. |
| R-011 | Risk | High | Humanoid controller failures may be confused with high-level goal infeasibility. Concretely observed in the ManiSkill3 spike (2026-07-28): a naive constant-hold action falls within ~0.5s even with zero injected disturbance — a controller-quality problem that would look identical to "infeasible" without careful separation. Concretely observed *again*, differently, in D-024/D-028: G1's arm genuinely cannot reach within contact range of two specific objects from any reasonable standing position — confirmed as a real kinematic limit (not a controller bug) only after building a proper analytic-Jacobian IK solver and searching broadly. Reinforces this risk's core concern: distinguishing "can't do it" from "control/tooling failed to do it" took real, non-trivial verification work both times. | Use a skill interface, repeated/oracle reachability labels, and separate error decomposition. |
| R-012 | Risk | Medium | Humanoid simulation and visual RL may exceed the compute budget. Partially confirmed: no CUDA on the primary dev machine, so GPU-vectorized parallel sim isn't available there — CPU sim is workable for single-env dev only. | Prototype logic cheaply, reuse low-level skills, freeze encoders initially, retain humanoid as the final gate, and budget for a CUDA machine/cloud GPU before any parallel RL training phase. |
| R-013 | Risk | High | Confirmed, open, unfixed upstream ManiSkill3 rendering bug (D-022, `haosulab/ManiSkill#1150`): rendered frames from the real-scene envs desync from the actual scene after roughly the second render-producing reset in one process (macOS, YCB-object scenes specifically). Not fixable in this project — it's in a dependency. Currently mitigated with a runtime warning guard and by keeping clip_feasibility.py's/dinov2_probe.py's own tests inside the verified-safe budget (≤2 in-process renders, or subprocess-isolated capture for more). Real risk if this project ever needs bulk/batch rendering (e.g. generating a large visual dataset) on this platform. | Check whether `haosulab/ManiSkill#1150` has a fix in a future ManiSkill3 release before attempting any batch-rendering workflow on macOS; budget for a Linux/CUDA machine as a fallback if it doesn't. |
| R-014 | Risk | Medium | Object-existence/position state for `tidy_up_env_replicacad_humanoid.py`'s scene builder may not be reliable for a *new* `build_config_idx`, even with D-021's existing torch-seed pinning. Confirmed 2026-08-06 (D-061, not fixed): a candidate third scene layout, extensively validated standalone (15+ runs, multiple independent script structures, all agreeing), deterministically disagreed with the real registered `scene_variant` code path once wired in (different object positions, target objects hidden) — 15/15 identical wrong results, not flaky. The two already-shipped layouts (`kitchen_cabinet`/`kitchen_sink`) are confirmed robust by extensive prior use; the actual mechanism making a *new* index unreliable was not isolated despite ruling out seed, `PYTHONHASHSEED`, import order, `env.step()` calls, and cross-instantiation ordering (D-022's known class of bug) as the cause. | Before trusting any *new* `build_config_idx` for this scene builder, validate object placement through the exact real `scene_variant` registration path (not a standalone/patched harness) with a large, repeated sample — standalone validation alone was not sufficient evidence here. Investigate the ManiSkill3 rearrange scene builder's actual object-visibility-assignment code directly before the next attempt, rather than more black-box trial and error. |

**R-010 update (D-091, 2026-08-12):** the execution-contract remainder in
the R-010 row is now closed. `_navigate_to()` screens the actual planned
waypoints or direct fallback before driving; a rejected route stops with zero
motion and returns the guard reason as a safety skip. Stop-and-report is
deliberate until the planner supports constraint-aware alternate-route search.
The remaining approximation is swept spheres/points rather than full robot-link
collision geometry.

**R-010 follow-up (D-092, 2026-08-12):** constraint-aware alternate-route
search is now implemented. Predicted affected objects are inflated into a copy
of the occupancy grid, the detour is replanned and independently re-screened,
and D-091's zero-motion stop remains the fallback. The remaining approximation
is still 2D spherical clearance rather than full robot-link geometry.

**R-010 result (D-097, 2026-08-13):** a live safety-matched counterfactual
now directly addresses the original “safe by doing nothing” concern. Stop-only
preserved the protected object but skipped an achievable goal; replanning kept
the same exact zero displacement and completed it. Remaining scope limitation:
one controlled geometry, not a distribution of naturally occurring hazards.

**R-010 robustness follow-up (D-098, 2026-08-13):** the positive live result
holds with the hazard at 30%, 50%, and 70% of the original route: three goal
completions and exact zero protected-object displacement. This removes the
single-location caveat, but remains one route, object, scene, and seed rather
than a distribution of naturally occurring hazards.

**R-010 second-route follow-up (D-099, 2026-08-13):** the same three-location
result holds on the bowl route. Across both goal routes and six controlled
hazard locations, every goal completes with exact zero protected-object
displacement. Remaining scope: one protected-object type, scene, and seed.

**R-010 semantic-object follow-up (D-100, 2026-08-13):** an alternate valid
graph protecting `cracker_box` produces the same live result: detected from
the constraint, safely bypassed, goal completed, exact zero displacement. This
removes the single-object/hardcoding caveat. Remaining live scope: one
scene/layout and seed.

**R-010 ablation-gating regression, found and fixed (D-105/D-106,
2026-08-13):** running the full suite against D-091–D-104 together for the
first time (each of those decisions individually skipped it) found that
D-091's unconditional navigation-level safety screening had silently broken
D-058's original unguarded ablation — the specific test that proves the
guard does real work, not vacuously. `use_intent_guard=False` disabled the
high-level `validate_action()` check but not the independent navigation-level
one, so an "unguarded" run could no longer produce a real violation at all.
Fixed by threading the same flag through to a new, opt-in
`enable_safety_screening` parameter (D-105) — restoring a genuine
zero-protection baseline. A real, generalizable lesson for this risk
specifically: any *new* safety mechanism added later must also be included in
what this ablation disables, or the same silent gap reproduces. Fixing the
gating alone still didn't make the test pass, surfacing a second, unrelated
issue: the test's original protected object, `master_chef_can`, turns out to
be structurally unreachable by Fetch from spawn in this scene (confirmed via
grid connected-components analysis, not a discretization artifact — persists
across resolutions from `0.15` down to `0.05`). Invisible everywhere else in
the project because every other live navigation test only ever routes
*around* `master_chef_can`, never *to* it. Swapped to the already-reachable
`cracker_box` (D-100's own alternate-object pattern) rather than weaken the
test's claim (D-106).

**R-010 geometry fix (D-101, 2026-08-13):** mobile screening now projects
object centers onto the XY path plane, fixing the prior false negative for a
floor-level object directly in the base corridor. Verified in pure height-
invariance/negative-control tests and one real Fetch detour at `z=0.05`.
Remaining approximation: circular 2D clearance rather than full robot-link and
object-footprint geometry.

**R-010 extent follow-up (D-102, 2026-08-13):** production screening now
derives object radii from real SAPIEN convex collision vertices/scale/local
pose, closing the point-object half of the approximation. A live mesh overlap
was detected with its center outside the old threshold. Remaining approximation:
Fetch motion is still one constant circular clearance footprint rather than
full robot-link geometry.

**R-010 robot-footprint follow-up (D-103, 2026-08-13):** Fetch's real
circumscribed base radius is ≈0.288 m, but adopting that rotation-invariant
circle by default caused 6 previously successful live detours to become
fail-closed stops. It remains an explicit ablation; production retains the
empirically validated 0.2 m clearance. Remaining need: oriented base/full-link
geometry that improves safety fidelity without this avoidable recall loss.

**R-010 execution-validity correction (D-104, 2026-08-13):** live goal credit
now requires measured base arrival. This exposed and fixed a missing final
approach segment after the grid planner's nearest-free target cell. Failed
navigation can no longer earn completion through unconditional teleportation;
the corrected real detours reach a measured `0.65 m` manipulation standoff
before manipulation. Intermediate waypoint acceptance is also capped at
`0.2 m`, preventing the controller from cutting across several `0.15 m` grid
cells. Sixteen affected focused tests pass.

**R-010 multi-seed benchmark (D-108, 2026-08-13):** the reachable-target side
of the navigation-safety stack was checked under real seed variance for the
first time (previously only hand-placed single scenarios). Swept candidate
`onset_step_range` windows and found the existing single-seed regression
test's window degenerate (zero wasted steps every seed); a wider window
produces real cross-seed variance. Across 30 paired seeds (`static` vs
`oracle_feasibility`, `bowl_destroyed`): identical `goals_achieved`
seed-for-seed, and a paired-bootstrap-significant reduction in wasted steps
for `oracle_feasibility` (independent CIs overlap; the paired per-seed
difference does not). New regression test passes standalone (259.83s).

**R-010 unreachable-route correction (D-107, 2026-08-13):** a failed
collision-aware grid plan no longer falls back to a straight-line production
drive. It returns an explicit geometric failure with zero motion; only the
named unguarded research ablation retains direct driving.
The real ReplicaCAD target identified in D-106 exercises this branch
end-to-end with zero control steps and exactly zero base-position change
(D-109), rather than relying only on a mocked planner result. D-112 repeated
the full attempt/aggregation contract on seeds 0, 1, and 2 with identical
zero-motion results. D-113 swapped a second object identity into the same
region and reproduced the result, ruling out an object-name special case;
D-114 reproduced it in a second disconnected occupancy component. Scene-layout
diversity remains open because the Fetch env exposes only one pinned build
configuration and R-014 still blocks trusting a newly added one.

## Resolved or superseded

| ID | Resolution date | Resolution |
|---|---|---|
| R-001 | 2026-07-26 | Original asset-import risk superseded by R-011/R-012 and the new skill-interface design. |
| R-004 | 2026-07-26 | Replaced old injected-failure concern with intervention validity and leakage risks. |
| I-002 | 2026-07-26 | Original model-selection issue superseded by the broader humanoid simulator/asset decision in I-003. |
| I-003 | 2026-08-02 | Formally selected ManiSkill3 as the primary humanoid-capable simulator (D-033) instead of spiking Isaac Lab for a head-to-head. D-006 required a spike step, not necessarily a second candidate, and the evidence bar was already met: seven weeks of continuous ManiSkill3-specific work (D-009 through D-032) with nothing disqualifying found, against one known and worked-around platform gap (`mplib` on Apple Silicon). An Isaac Lab spike remains a real option later (e.g. if a specific ManiSkill3 limitation blocks something), but isn't owed by D-006 and wasn't worth the cost of a second full simulator integration for a Low-severity open question with this much evidence already in hand. |
| I-000 | 2026-07-24 | Project scope set to simulation-only. |
| I-004 | 2026-08-06 | Resolved (D-062): language backbone (`instruction_parser.py`, D-019/D-038) and SSL visual baseline (DINOv2) selected, both preconditions this row's own mitigation named (schema review, compute budget) having resolved (D-037, R-012). Not "CLIP loses" — CLIP stays the pipeline's real feasibility backend (zero-shot, no training data, generalized to a real distribution shift D-054 found with no extra work); DINOv2 is the committed self-supervised baseline H1's own comparative question needs, not a discarded alternative. See `ai-notes/model-comparison-clip-vs-dinov2.md` for the updated evidence this call was based on. |
