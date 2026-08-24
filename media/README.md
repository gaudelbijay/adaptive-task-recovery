# Media

`demos/` holds real captured episode GIFs referenced from the root
[`README.md`](../README.md) — real ManiSkill3 render output (subprocess-
isolated per D-022's rendering-desync guard, one reset per capture process),
not mockups or scripted camera moves. Regenerate with a fresh capture script
rather than hand-editing a GIF if the underlying behavior changes.

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
