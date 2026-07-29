# ManiSkill3 humanoid simulator spike

**What this is:** a Phase 0 spike evaluating ManiSkill3 as one candidate
simulator, per [docs/04-benchmark-environment.md](../../docs/04-benchmark-environment.md)
("Selection requirements", "Candidates should be compared through a small
spike rather than chosen from reputation") and
[D-006](../../ai-notes/decisions.md) ("Phase 0 includes a simulator spike. No
simulator-specific architecture should be committed before it passes the
selection criteria."). **This is investigation code, not the project's
benchmark environment or task schema** — it doesn't implement goals,
language, feasibility, or the `WorldIntervention` API.

## What it tests

Two things, in two separate env registrations:

1. **Humanoid loading + scripted physical events** (`humanoid_stand_spike.py`):
   can ManiSkill3 load a humanoid asset, run it stably enough to be usable,
   and support a deterministic seeded scripted event mid-episode? A standing
   humanoid gets one scripted external push at a random-but-seeded control
   step, registered as `HumanoidStandSpike-G1-v1` / `HumanoidStandSpike-H1-v1`.
2. **Object-level interventions** (`object_intervention_spike.py`) — the
   actual gating question for I-003/D-006, since the research question needs
   `WorldIntervention`-style object/scene changes, not physical pushes: can
   an object be removed/destroyed mid-episode, and can *new* geometry (a
   blocking obstacle) be added to an already-built scene mid-episode?
   Registered as `ObjectInterventionSpike-v1`. Uses ManiSkill3's plain
   `panda` arm, not a humanoid — object mechanics are embodiment-agnostic.
3. **Manipulation skill + RGB-D observations** (`manipulation_skill_spike.py`)
   — does ManiSkill3 support the "reusable reach/grasp" skill requirement,
   and does RGB-D observation mode actually work? Runs on ManiSkill3's
   built-in `PickCube-v1` task, not a custom env.

## Device-agnostic by design

All scripts/tests pick their sim backend via `device_utils.resolve_sim_backend()`,
which uses CUDA if `torch.cuda.is_available()` and CPU otherwise — **not**
ManiSkill3's own `sim_backend="auto"`, which doesn't check CUDA availability
at all (it only branches on `num_envs`, so a single-env run stays on CPU
under "auto" even with a GPU present). `humanoid_stand_spike.py`'s push
force-application branches internally between the CPU per-body API
(`add_force_at_point`) and the GPU batched-tensor API
(`cuda_rigid_body_force`) so the same code path works either way.
`object_intervention_spike.py` is the one deliberate exception: object
add/remove is unsupported under GPU-batched sim by SAPIEN's own design (fixed
per-actor buffers allocated at reconfigure time), so it always forces CPU
and raises a clear `RuntimeError` if that's violated, rather than failing
silently.

**Caveat:** this dev machine (Apple M4 Max) has no CUDA, so only the CPU
path has actually been run. The GPU branches are written to the same API
pattern ManiSkill3's own `Actor.apply_force` uses internally, but are
untested here — verify on a CUDA machine before relying on them.

## How to run it

Requires the `.maniskill` pyenv virtualenv (Python 3.12.12) — see
`requirements-maniskill.lock.txt` at the repo root for the exact working
version set (`mani-skill-nightly`, `sapien 3.0.2`, `torch 2.10.0`,
`gymnasium 1.2.3`).

```bash
pyenv activate .maniskill   # or: export PATH="$(pyenv root)/versions/3.12.12/envs/.maniskill/bin:$PATH"
pip install -e . --no-deps  # installs this spikes/ package
python -m pytest tests/spikes/ -v
python scripts/run_maniskill_humanoid_spike.py --robot g1 --episodes 5
```

## Findings (2026-07-28, Apple M4 Max, macOS, CPU sim backend)

Scored against [docs/04-benchmark-environment.md](../../docs/04-benchmark-environment.md)'s
selection requirements — ✅ tested and works, ⚠️ not exercised by this spike
(no evidence either way), ❌ found not to work:

| Requirement | Result |
|---|---|
| Humanoid model support | ✅ Unitree G1 and H1 both load and simulate out of the box — assets are bundled with the `mani_skill` pip package (G1) or one `download_asset` command away (H1, `mani_skill.utils.download_asset unitree_h1_simplified`). No manual URDF sourcing needed. |
| Deterministic seeding | ✅ Confirmed exactly reproducible: same seed → identical push onset step, severity, and force vector, across independent env instances and separate script runs. See `tests/spikes/test_maniskill_humanoid_spike.py::test_push_reproducible_given_seed`. |
| Privileged state for oracle labels | ✅ Exact simulator state (pose, qpos, contact) is directly readable — used for both the standing/fallen check and the push injection. |
| Controllable state transitions | ✅ **Now tested directly** (`object_intervention_spike.py`), not just the physical push. Confirmed: (a) an object can be genuinely removed from the live physics scene mid-episode (`actor.remove_from_scene()` — verified via low-level `scene.sub_scenes[0].entities` membership dropping, not just a Python-side flag), and (b) **new** geometry can be added to an already-built scene mid-episode (a "route blocker" actor spawned at a scripted step — entity count goes up by exactly one, confirmed collidable). Both are deterministic given a seed. This is the actual gating capability for I-003/D-006, and it works. |
| RGB observations | ✅ **Now tested** (`manipulation_skill_spike.py`): `obs_mode="rgbd"` on `PickCube-v1` returns real, non-degenerate `rgb` (128×128×3, uint8, full 0-255 range) and `depth` (128×128×1, int16, sensible mm-scale range) tensors. |
| Object-centric interaction, reusable nav/reach/grasp/place skills | ✅ **Partially confirmed** — reach + grasp + lift, specifically: a hand-scripted waypoint sequence using ManiSkill3's built-in Cartesian end-effector controller (`pd_ee_delta_pos`) picked up and lifted a cube 5/5 times across seeds 0-4 on `PickCube-v1`. Navigate/place and the *canned* motion-planning solutions were not tested (see gotcha below) — this confirms basic actuation/grasp mechanics work, not the full skill library. |
| Natural-language task generation | ❌ Not a ManiSkill3 capability — would need a custom instruction-generation layer regardless of simulator choice. |
| GPU vectorization | ⚠️ Code is written to use it automatically when available (see "Device-agnostic by design" below), but **untested** — this dev machine (Apple M4 Max) has no CUDA, so SAPIEN's GPU-vectorized PhysX backend is unavailable here. CPU backend works fine for single-env development at ~450–600 steps/sec. **Any future large-scale parallel RL training will need a CUDA machine** (cloud GPU) regardless — this is a shared-infra requirement, not specific to ManiSkill vs. Isaac Lab (Isaac Lab also requires an NVIDIA GPU, and more insistently). |

### Object-level intervention findings (2026-07-28)

- **Object removal is real, not cosmetic.** `actor.remove_from_scene()`
  genuinely removes the entity from the low-level PhysX scene — verified by
  entity count dropping and the entity no longer appearing in
  `scene.sub_scenes[0].entities`. This works in CPU sim only (matches the
  `Actor.apply_force`/removal docstring's own note that removal isn't
  supported in GPU sim mode).
- **Gotcha: the high-level `Actor` wrapper goes stale after removal.**
  `target.pose` and `target.px_body_type` keep returning their cached
  pre-removal values instead of erroring or reflecting removal — there is no
  "does this actor still exist" query on the wrapper itself. Any oracle/eval
  code (the "before/after state" logging the Intervention API needs) must
  track existence with its own flag at the moment of removal, not by
  re-querying the object afterward. See
  `InterventionRecord.exists_after` and
  `tests/spikes/test_object_intervention_spike.py::test_actor_wrapper_goes_stale_after_removal`.
- **New geometry can be added mid-episode**, not just at initial scene
  construction — confirmed by spawning a static "blocker" actor at a
  scripted step and seeing the physics-scene entity count increase by
  exactly one. This is the capability object removal alone doesn't prove,
  and it's what "a route becomes blocked" / "an obstacle appears" would rely
  on.
- Both interventions are deterministic given a seed, same as the push in the
  humanoid spike.

### Manipulation skill + RGB-D findings (2026-07-28)

- **RGB-D works cleanly**: `obs_mode="rgbd"` returns `sensor_data.<camera>.rgb`
  (uint8, 0-255) and `.depth` (int16, mm-scale) with real, non-degenerate
  content — confirmed on `PickCube-v1`.
- **Gotcha: `mplib` (ManiSkill3's motion-planning dependency) does not build
  on this machine.** `pip install mplib` fails — its build pins
  `libclang==11.0.1`, and no wheel exists for that exact version on
  Python 3.12 / macOS arm64 (only 0.2.0/0.2.1 exist on PyPI, both pin the
  same broken constraint). This blocks the *canned* motion-planning
  solutions ManiSkill3 ships (`mani_skill.examples.motionplanning.panda...`)
  — worth knowing before assuming those demos "just work" on Apple Silicon.
- **Workaround that does work**: ManiSkill3's built-in Cartesian
  end-effector controller (`pd_ee_delta_pos`) uses `pinocchio` for IK, not
  `mplib` — and `pinocchio` installs cleanly via `pip install pin` (native
  arm64 wheel, no build-from-source). A simple hand-scripted waypoint
  sequence (no collision-aware path planning) using this controller reached,
  grasped, and lifted the cube 5/5 times. Good enough to confirm basic
  actuation/grasp mechanics; a *collision-aware* path planner would still
  need either a working `mplib` build or a different IK/planning approach.

**Bonus finding, relevant to [R-011](../../ai-notes/issues_and_risks.md):**
under a naive constant-hold action (no trained/tuned balance controller), the
humanoid falls within ~0.5s (~28–31 of 200 steps) *even with zero push
force*. This is concrete evidence for R-011's concern — a low-level
controller failure here would be trivially indistinguishable from "goal
infeasible" if a future evaluation harness didn't separate the two. A real
standing/balance controller (trained or hand-tuned) is separate future work,
not something this spike attempted.

Raw results: `results/maniskill_humanoid_spike_{g1,h1}.json` (gitignored —
regenerate with the command above). Videos:
`results/videos/maniskill_humanoid_spike_{g1,h1}/{baseline_no_push,with_push}/*.mp4`,
`results/motionplanning_spike/*.mp4`.

## Bottom line

ManiSkill3 clears every requirement tested so far: humanoid support,
deterministic seeding, privileged state, object-level interventions (removal
+ mid-episode geometry addition), RGB-D observations, and basic reach/grasp
manipulation. It's workable on non-CUDA dev hardware, with one real caveat —
the `mplib`-dependent canned motion-planning solutions don't build on Apple
Silicon macOS, though a `pinocchio`-based IK controller works as a
substitute. Still open: natural-language task generation (expected — not a
simulator capability at all) and an Isaac Lab spike for comparison. Per
D-006, this is now substantial evidence for ManiSkill3, though the choice
technically remains open until Isaac Lab gets an equivalent look.
