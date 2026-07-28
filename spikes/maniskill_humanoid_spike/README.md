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

One narrow question: can ManiSkill3 load a humanoid asset, run it stably
enough to be usable, and support a deterministic seeded scripted event
mid-episode, on this dev machine? Concretely: a standing humanoid gets one
scripted external push at a random-but-seeded control step
(`scripted_intervention.py`), registered as two gym envs
(`HumanoidStandSpike-G1-v1`, `HumanoidStandSpike-H1-v1`).

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
| Controllable state transitions | ⚠️ Only tested one narrow case: a scripted *physical* force push. The project's actual `WorldIntervention` needs (object removed/broken, route blocked, resource consumed — see docs/04 "Candidate irreversible changes") are object/scene-level, not physical-force-level, and are **not validated by this spike**. |
| RGB observations | ⚠️ Not exercised — this spike used `obs_mode="state"` throughout. ManiSkill3 natively supports RGB-D observation modes and was used here only for `render_mode="rgb_array"` video capture, which does work. |
| Object-centric interaction, reusable nav/reach/grasp/place/safe-stop skills | ⚠️ Not exercised — this spike's task is pure standing on an empty ground plane, no objects. ManiSkill3 ships example manipulation tasks (`PickCube-v1`, `PushCube-v1`, `OpenCabinetDoor-v1`, etc. — 74 registered envs total in this install) suggesting a skill library exists, but none were run here. |
| Natural-language task generation | ❌ Not a ManiSkill3 capability — would need a custom instruction-generation layer regardless of simulator choice. |
| GPU vectorization | ❌ on this machine specifically: no CUDA (Apple M4 Max has no NVIDIA GPU), so SAPIEN's GPU-vectorized PhysX backend is unavailable. CPU backend (`sim_backend="cpu"`) works fine for single-env development at ~450–600 steps/sec. **Any future large-scale parallel RL training will need a CUDA machine** (cloud GPU) — this is a shared-infra requirement, not specific to ManiSkill vs. Isaac Lab (Isaac Lab also requires an NVIDIA GPU, and more insistently). |

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

ManiSkill3 clears the humanoid-support, deterministic-seeding, and
privileged-state requirements convincingly, and is workable on non-CUDA dev
hardware. It has **not** been evaluated against the requirements that matter
most for the actual research question — object-level interventions,
language, and the reusable skill library — nor against the other candidate
(Isaac Lab). Per D-006, this alone should not be read as "ManiSkill selected."
