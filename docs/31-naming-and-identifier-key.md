---
title: Naming and Identifier Key
status: active
last_updated: 2026-09-01
---

# Naming and identifier key

This repository accumulated several independent numbering schemes that use the
same tokens for different things. This document is the canonical key. It exists
because the counters are load-bearing for provenance and cannot simply be
deleted, but they are not self-describing and have collided.

## The rule

**Prose names a method by what it does. A counter appears only when it
identifies a frozen artifact**, and then it is qualified by its series.

A counter is *not* a quality ordering. A higher number is a later candidate,
not a better one, and in every series most candidates were rejected. Writing
"V19 beats V10" is meaningless: those belong to different series and are not
comparable.

## The series

| Series | Range | What it counts | Where it lives |
|---|---|---|---|
| Environment generation | v1–v4 | Registered ManiSkill env + reward semantics | `LearnedRecovery-v{1,2,3,4}` gym ids |
| Preregistered visual hypotheses | V1–V5 | **Hypotheses**, not models | `docs/16-visual-recovery-hypotheses.md` |
| Visual recovery controllers | V1–V60 | RGB policy candidates | `docs/14`, `docs/17` |
| Recovery router candidates | V1–V10 | Router / dispatch candidates | `docs/30`, `configs/a_plus_recovery_gate_v*.json` |
| External Peg gates | v1–v5 | Preregistered Peg gate revisions | `configs/a_plus_external_peg_insertion_gate_v*.json` |
| Peg nominal PPO | v2–v9 | Peg nominal controller configs | `configs/external_peg_nominal_ppo_v*.json` |
| Router checkpoint sets | v1–v18 | Offline router training runs | `results/router/v*_*/` |
| Peg offline dataset | v1, v2 | Prefix dataset revisions | `results/router/external_peg_offline_development_v*` |

## Known collisions

These are real ambiguities in existing text. When reading an older document,
resolve the token by its series, not by the number.

- **V3** is at least three things: environment generation `LearnedRecovery-v3`,
  visual hypothesis V3 (temporal representation learning), and router candidate
  V3 (full-geometry centering). `docs/16`'s hypothesis V5 refers to "the matched
  V3 adaptive state PPO", where that V3 is the *environment*, not the hypothesis.
- **V28** is two things: in `docs/30` and the README it is the hand-written
  motion-threshold hybrid controller, tagged from its `280000000` confirmatory
  seed family; in `docs/17` it is the 28th visual candidate, paired
  rendered-domain distillation.
- **V19** is the dual-specialist RGB controller in every document, but it also
  appears as a *prefix* in `docs/17` row titles such as "V19 paired
  rendered-domain distillation V28", where the first token names the frozen base
  policy and the second names the candidate built on it.
- `docs/30` mixes series in one file: V1–V10 are router candidates, while V19
  and V28 in the same prose are visual-series artifacts.

## Descriptive names

Use the right-hand column in prose. The counter stays available for tracing back
to the frozen artifact.

### Recovery router candidates (`docs/30`)

| Tag | Name | Outcome |
|---|---|---|
| V1 | instantaneous-geometry router | rejected: mechanism identifiable from one state |
| V2 | temporal-composition router | rejected: incomplete geometry centering |
| V3 | full-geometry centered router | rejected: nominal below the worst-condition floor |
| V4 | nominal-state controller revision | rejected on selection |
| V5 | selected-nominal router | rejected by one violation (3.0208% vs 3.00%) |
| V6 | option-specific nominal router | rejected: nominal retention |
| V7 | evidence-release router | rejected: nominal retention |
| V8 | terminal-ensemble router | rejected: gain 4.51 pp below the 5 pp floor |
| V9 | reverse-handoff router | rejected: pooled OOD 69.64% |
| **V10** | **guarded factorized dispatch router** | **passed its once-only confirmation** |

### Visual controllers referenced across documents

| Tag | Name |
|---|---|
| V6 | clean learned-progress DAgger RGB controller |
| V7 | adaptive continuation controller |
| V11 | strict-trained state PPO (strict specialist) |
| V12 | integrated-mixture state PPO |
| V13 | stabilized integrated RGB controller |
| **V19** | **dual-specialist RGB controller** (integrated incumbent) |
| V20 | full-strength VICReg ablation |
| V21 | low-variance VICReg extension |
| V35 | observed-domain canonicalization controller |
| V41 | magnitude-gated canonicalization controller |
| V60 | residual-corrector composition controller |

### Control ladder rungs (`scripts/audit_shortcut_ladder.py`)

These have no counters and should not acquire any.

| Rung | Name |
|---|---|
| 1 | instantaneous control (current frame only) |
| 2 | one-past-frame control |
| 3 | hand-written motion rule |
| 4 | recurrent model (factorized / unstructured) |

## What must not be renamed

These are frozen identifiers with provenance attached. Renaming them breaks
reproducibility of an already-opened confirmation.

- Registered gym ids (`LearnedRecovery-v4`, `PegInsertionSide-v1`).
- Preregistered gate config filenames and their SHA-256 digests. These carry an
  `a_plus_` prefix — an early internal label for "the bar a result had to clear",
  nothing to do with the result it produced. The prefix is meaningless as a
  quality signal and several of the gates it names were failed; it survives only
  because the filenames are hashed into opened confirmations and the manifests
  key on them. The standard those gates actually encode is in
  [`19-evidence-standards.md`](19-evidence-standards.md).
- The persisted `model` string inside a checkpoint (`"causal_gru"`), which gate
  manifests, `router_checkpoint_sha256` provenance, and audit scripts key on.
  See `src/atr/policies/option_router.py`.
- Seed family numbers (`347000000`, `425000000`, `429000000`).
- Slurm job ids and result directory names already written to disk.

## Going forward

New candidates get a descriptive name first and a counter only if they produce
a frozen artifact. If a new numbering scheme is unavoidable, add it to the
series table above in the same commit.
