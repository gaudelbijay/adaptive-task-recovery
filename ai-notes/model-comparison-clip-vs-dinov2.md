# Measured comparison: zero-shot CLIP vs. DINOv2 linear probe

**Status:** evidence toward I-004 (`ai-notes/issues_and_risks.md`), **not a
formal selection.** I-004's own mitigation note says selection should wait
until the task schema and compute budget are known, and D-013's schema is
still out for teammate review (`ai-notes/review-request-task-schema.md`).
This document exists so that decision, whenever it's made, has real
numbers behind it instead of vibes — same spirit as `Goal.condition` (D-026)
being flagged PROPOSED rather than smuggled in as settled.

**Scope:** both models are already implemented and validated for this
project's narrow task (`spikes/task_schema_draft/clip_feasibility.py`, D-020;
`spikes/task_schema_draft/dinov2_probe.py`, D-023). This document adds the
measurements `docs/08-training-pipeline.md` says a model selection needs
that weren't recorded anywhere yet: latency, memory, and licensing,
plus one direct calibration measurement. Accuracy/generalization/downstream-
utility findings below are pulled from existing decisions (D-020, D-023,
D-027, D-029), not re-derived here.

## What each model actually is, in this project

- **CLIP** (`open_clip`, `ViT-B-32-quickgelu`, OpenAI pretrained weights):
  zero-shot. A hand-picked crop region + a hand-picked text prompt per
  object; no training data, no fine-tuning, judgment is a thresholded
  image-text similarity margin.
- **DINOv2** (`facebookresearch/dinov2`, `dinov2_vits14`): self-supervised,
  no text/label supervision at all. Needs a small labeled dataset (a
  handful of present/absent crops) to fit a logistic-regression linear
  probe on top of its frozen embeddings before it can judge anything.

That difference — zero-shot vs. probe-fitting — drives most of the
integration-cost gap below; it's not incidental.

## Latency and memory (measured, this machine)

CPU only (Apple M4 Max, macOS, no CUDA — this project's only current dev
hardware, see R-012). Each model benchmarked in its own subprocess to get
a clean peak-RSS reading; 20 warmed-up inference calls averaged per model,
on a representative 200×200 crop.

| | CLIP (ViT-B-32-quickgelu) | DINOv2 (ViT-S/14) |
|---|---|---|
| Parameters | 151.3M | 22.1M |
| Model load time | ~1.7–2.3s | ~0.4s |
| Peak RSS after load (delta from baseline) | ~1287 MB | ~178 MB |
| Inference latency (mean ± stdev, 20 calls) | 33.0 ± 0.9 ms | 15.2 ± 0.4 ms |

DINOv2 ViT-S/14 is ~6.9x fewer parameters, ~7x lower memory delta, and
~2.2x faster per call than CLIP ViT-B-32 here — expected, since ViT-S is a
smaller architecture than ViT-B and CLIP's call also encodes two text
prompts every time (not cached across calls in the current
`clip_margin()` implementation — a real implementation inefficiency, not
an inherent CLIP cost, since the prompts are fixed per object/scene and
could be pre-encoded once).

Neither number is a bottleneck at this project's current toy scale (one
decision per goal, a handful of goals per episode). Would matter more if
the benchmark ever needs per-frame feasibility judgments at real control
rates.

## Calibration (measured, one direct run)

CLIP's current interface (`clip_margin()`) returns a raw cosine-similarity
margin, thresholded at 0 — not a probability. There is nothing to compute
a calibration metric against without changing that interface, so no CLIP
calibration number exists; recorded here as a finding, not filled in with
a guess.

DINOv2's probe already outputs `predict_proba` (unused by
`fit_and_evaluate_probe()`, which only returns hard predictions). Ran the
same leave-one-out procedure directly against `predict_proba` for one
6-present/6-absent set (`master_chef_can`, `kitchen_cabinet`, matching
`tests/drafts/test_dinov2_probe.py`'s scale): **100% LOO accuracy, Brier
score 0.0001** — every held-out "present" example got predicted probability
~0.99, every "absent" example ~0.01, regardless of which specific example
was held out. That near-zero Brier score should not be read as "DINOv2 is
well-calibrated" in general: `dinov2_probe.py`'s own module docstring
already flags that every example in this pinned-layout setup is visually
almost the same scene (D-021), so the embeddings within each class are
close to duplicates — the probe finding a wide, confident margin is
consistent with easy separability at this scale, not evidence of
calibration under real visual variation. A real calibration test needs
held-out layouts/lighting/objects, which doesn't exist yet for either
model.

## Licensing (verified against source, not assumed)

| | License | Source |
|---|---|---|
| CLIP (`open_clip` code) | MIT | `mlfoundations/open_clip` LICENSE |
| CLIP (OpenAI pretrained weights, `pretrained="openai"`) | MIT | `openai/CLIP` LICENSE (weights ship from the same repo/license) |
| DINOv2 (code + standard backbone weights, incl. ViT-S/14) | Apache License 2.0 | `facebookresearch/dinov2` LICENSE |

Both are permissively licensed for this project's use (research, and MIT/
Apache 2.0 both permit commercial use too, if that ever mattered). **Not**
a differentiator between them — worth having checked rather than assumed,
since DINOv2's *original* 2023 release used a more restrictive
non-commercial license before Meta relicensed it; that history is easy to
misremember as still current.

## Accuracy, generalization, downstream utility (existing evidence, cited not re-run)

| | CLIP | DINOv2 |
|---|---|---|
| Matches oracle feasibility | 4/4 cases, `kitchen_cabinet` (D-020); 2/2 further cases, `kitchen_sink` (D-027) | 100% LOO accuracy, 8→20 examples grown over D-023/D-026, `kitchen_cabinet` only |
| Scene layouts validated | 2 (`kitchen_cabinet`, `kitchen_sink`) | 1 (`kitchen_cabinet`) — `kitchen_sink` support exists in code (D-027) but has no DINOv2 test/result against it yet |
| Wired into the live decision loop | Yes — `end_to_end.py` (D-029) uses it for real, matching oracle end-to-end | No — deliberately not wired in (D-029's "Consequences": needs a pre-fit probe from multiple examples, not a single-frame judgment; kept as a separately-validated alternative backend |
| Needs labeled data to operate | No (zero-shot) | Yes (a probe fit from labeled present/absent crops per object) |

## Integration cost (qualitative, not measured — a real axis docs/08 asks for anyway)

- **CLIP**: add a new object → hand-pick a crop region and a text prompt
  (found by trial: generic templates measurably underperformed
  brand-specific ones, per `clip_feasibility.py`'s own docstring). No
  labeled data collection needed. Fails immediately and loudly
  (`ValueError`) on an uncalibrated object/scene — no silent wrong
  answers.
- **DINOv2**: add a new object → collect labeled present/absent crops
  (here, constrained further by D-022's rendering bug into one subprocess
  per example — `capture_episode_subprocess.py`), then fit and validate a
  probe. More steps, more infrastructure, but the resulting judgment
  doesn't depend on hand-writing a text prompt that happens to work.

## Bottom line

Neither model dominates on every axis, which is exactly why this stays
evidence-for-a-future-decision rather than a selection:

- CLIP: zero training data, already wired into the real decision loop,
  fails loudly on unknown inputs, validated on 2 scene layouts — costs
  more compute per call and needs a hand-tuned prompt per object.
- DINOv2: ~7x cheaper per call, no text-prompt engineering, matches CLIP's
  accuracy on the one scene it's been tested on — needs labeled data to
  operate at all, more constrained data-collection process (D-022), and
  one fewer scene layout's worth of validation.

Revisit once D-013's schema review resolves and the benchmark's actual
scale (episode count, real-time constraints, how many objects need
calibration) is known — that's what determines whether the latency/memory
gap above is decisive or irrelevant.
