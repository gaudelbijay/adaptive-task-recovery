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
| RGB observations | ⚠️ Not exercised — this spike used `obs_mode="state"` throughout. ManiSkill3 natively supports RGB-D observation modes and was used here only for `render_mode="rgb_array"` video capture, which does work. |
| Object-centric interaction, reusable nav/reach/grasp/place/safe-stop skills | ⚠️ Object *existence/removal* mechanics are now tested (above), but not a reusable skill library — no navigate/reach/grasp/place was exercised. ManiSkill3 ships example manipulation tasks (`PickCube-v1`, `PushCube-v1`, `OpenCabinetDoor-v1`, etc. — 74 registered envs total in this install) suggesting a skill library exists, but none were run here. |
| Natural-language task generation | ❌ Not a ManiSkill3 capability — would need a custom instruction-generation layer regardless of simulator choice. |
| GPU vectorization | ❌ on this machine specifically: no CUDA (Apple M4 Max has no NVIDIA GPU), so SAPIEN's GPU-vectorized PhysX backend is unavailable. CPU backend (`sim_backend="cpu"`) works fine for single-env development at ~450–600 steps/sec. **Any future large-scale parallel RL training will need a CUDA machine** (cloud GPU) — this is a shared-infra requirement, not specific to ManiSkill vs. Isaac Lab (Isaac Lab also requires an NVIDIA GPU, and more insistently). |

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
`results/videos/maniskill_humanoid_spike_{g1,h1}/{baseline_no_push,with_push}/*.mp4`.

## Bottom line

ManiSkill3 clears humanoid-support, deterministic-seeding, privileged-state,
and — now confirmed — **object-level intervention support** (both removal
and mid-episode addition of new geometry), the requirement that actually
mattered most for the research question. It's workable on non-CUDA dev
hardware. Still open: RGB/language integration and the reusable skill
library are untested, and Isaac Lab hasn't been spiked at all — so per
D-006, this is strong evidence, not yet a selection.
