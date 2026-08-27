# Media

`demos/` holds real captured episode GIFs referenced from the root
[`README.md`](../README.md) — real ManiSkill3 render output (subprocess-
isolated per D-022's rendering-desync guard, one reset per capture process),
not mockups or scripted camera moves. Regenerate with a fresh capture script
rather than hand-editing a GIF if the underlying behavior changes.

`learned-recovery-montage.gif` is the Linux/CUDA exception to the older YCB
capture rule above: `capture_learned_recovery_policy.py` searches a declared
seed range without rendering, deterministically replays only the qualifying
seed, and validates safe success for each of the two intervention branches and
the nominal branch. JSON provenance and source MP4s live under
`results/learned_recovery/videos/`; `build_recovery_montage.py` only resamples
and labels those recordings.

`results/learned-recovery-v6-curves.{png,pdf}` is generated from the immutable
V6 per-seed metrics and held-out aggregate by `plot_manipulation_results.py`.
Both raster and vector exports are versioned for README/paper reuse.

The rest of this directory predates that and is unrelated:

## Architecture diagram status

**Superseded — see
[`docs/03-system-architecture.md`](../docs/03-system-architecture.md) for
the current, authoritative architecture diagram (D-035, 2026-08-02).**

`architecture-diagram.drawio` and its rendered preview files below
describe the superseded humanoid failure-detection and recovery
architecture — they predate the 2026-07-26 research reframing by a day
and were never updated to match it. Kept here only as design history;
not linked from the README, and not accurate to the current system.

The replacement diagram lives directly in `docs/03-system-architecture.md`
as a Mermaid diagram instead of a separate `.drawio` + rendered-image
pair: it renders natively on GitHub, stays diffable and version-controlled
as plain text, and can't drift out of sync with the prose next to it the
way a binary export can.
