---
title: Project Overview
status: draft
last_updated: 2026-08-01
---

# Adaptive Task Recovery

## One-liner

A feasibility-aware reinforcement learning agent that recognizes which
instructed goals survive an irreversible world change and adapts its strategy to
maximize valid goal completion without betraying the instruction.

## Core research question

A robot gets an instruction with more than one goal. Partway through, the
world changes in a way that can't be undone — something breaks, disappears,
or a path closes. Can the robot tell which goals are still possible, still do
whichever of them remain achievable, and never fake success by doing
something it was never asked to do?

> Can a reinforcement learning agent, conditioned on a parsed goal
> specification and equipped with learned visual representations, identify
> which goals remain feasible after unforeseen and irreversible world changes,
> and adapt its task strategy to maximize goal achievement without violating
> the original intent?

**What "language" means in this project, precisely.** No language model is used
in the policy loop. `src/atr/language/instruction_parser.py` compiles
controlled-grammar English into a `GoalGraph` offline; it is a hand-written
grammar, not learned, and its own module docstring says so. What reaches a
policy is a two-dimensional encoding of the resulting goal order, written
`instruction.0` and `instruction.1` in the feature schema. Describing these
policies as "language-conditioned" or "vision-language" overstates the input:
they are conditioned on a parsed goal-order encoding.

The one language-supervised model in the repository is CLIP
(`src/atr/feasibility/clip_feasibility.py`, open_clip ViT-B-32), used zero-shot
with text prompts for object presence in the earlier teleport-executor track.
It is not used by the router, the shortcut ladder, or any result in the
README.

## Build-up order (one capability at a time)

1. **Done already, zero perception or learning.** Hand-written goals, a
   privileged-state feasibility oracle, and an intent guard
   (`spikes/task_schema_draft/`, D-013 through D-018).
2. **Language.** Parse an actual instruction sentence into the goal graph,
   instead of writing one by hand.
3. **Vision, simplest version first.** Replace the privileged-state oracle
   with a feasibility judgment from images, starting with any working
   pretrained visual model.
4. **Self-supervised representations.** Swap in a representation learned
   from unlabeled data.
5. **RL policy.** Replace the scripted/oracle policies with one that's
   actually learned.
6. **Combine and evaluate end-to-end.**

Full phase breakdown: [`11-roadmap-and-milestones.md`](11-roadmap-and-milestones.md).

## Motivation

Long-horizon embodied agents usually assume that the world remains compatible
with their initial plan. In practice, an object may break, disappear, become
inaccessible, or be consumed; a passage may close; or an action may permanently
change another goal's feasibility. Continuing the original plan wastes effort,
while maximizing raw reward can produce a superficially successful action that
violates a constraint or changes the instruction's meaning.

ATR treats adaptation as three linked problems:

1. **Represent the changed world** from pixels using self-supervised features.
2. **Estimate goal feasibility** for each language-specified goal and constraint.
3. **Adapt strategy under intent constraints**, including justified partial completion.

## Formal view

An instruction is represented as goals `G = {g1, ..., gn}`, hard constraints
`C`, and optional priorities or preferences `P`. After an unannounced,
persistent intervention changes latent world state `z` to `z'`, the agent
observes pixels and history, estimates `Pr(feasible(gi) | o<=t, instruction)`,
and selects actions maximizing weighted valid goal achievement subject to `C`.
It must not obtain reward by silently redefining a goal.

## Scope (v1)

- Simulation-only, visually observable, object-centric tasks executed by a humanoid
- A humanoid-capable simulator, model, and library of reusable whole-body skills
- Natural-language instructions containing multiple goals and constraints
- Exogenous and action-induced irreversible changes
- Self-supervised visual pretraining or adaptation from unlabeled observations
- Held-out objects, layouts, paraphrases, and intervention types

The feasibility and intent components are designed to be embodiment-agnostic,
but v1 must include evaluation on a simulated humanoid. A simpler arm or abstract
environment may be used as a debugging testbed, not as the final evidence.

Out of scope for v1: real-robot deployment, training low-level dynamic locomotion
from scratch, unrestricted natural-language dialogue, open-web knowledge,
irreversible changes that cannot be visually or historically inferred, and
claims of general human-value alignment.

## Conceptual pipeline

```text
pixels + history --> self-supervised visual encoder --> world representation
language instruction --> goal/constraint encoder ------------------+
world representation + encoded goals --> feasibility estimator     |
                                                                   v
                                                  adaptive RL policy/planner
                                                           |
                                                intent guard/action mask
                                                           |
                                                        action
                                                           |
                                        environment + irreversible changes
```

## Known technical downsides, in plain language

- **Reinforcement learning is data-hungry and unstable.** It typically needs
  many thousands of trial-and-error attempts to learn anything, has no
  guarantee it converges to a good policy, and long tasks make credit
  assignment hard — if the agent fails, it's genuinely difficult to tell
  whether the bad decision was step 3 or step 30. Reward functions are also
  easy to get subtly wrong in ways that get gamed rather than solved.
- **Self-supervised visual representations aren't guaranteed to capture what
  the task needs.** They're trained with no notion that "feasibility" or
  "goals" exist — the encoder just learns whatever regularities are easiest
  to find in raw pixels. Whether it happens to encode "is this object still
  intact" in a way that's easy to read back out only shows up after
  training, by probing it directly.
- **Errors compound across the pipeline.** Vision feeds the feasibility
  model, which feeds the policy. A slightly-off visual representation
  produces a noisy feasibility signal, which produces a policy trained on
  bad labels, and a downstream failure is hard to trace back to the
  upstream piece that caused it. The roadmap keeps a privileged-state oracle
  as a stand-in and swaps in each learned piece one at a time (Phase 3–5 in
  `11-roadmap-and-milestones.md`) specifically to keep failures isolated.
- **It's resource-hungry.** RL usually needs many parallel simulated
  episodes running at once, and self-supervised pretraining usually needs a
  large volume of unlabeled data. The current dev machine has no CUDA GPU
  (D-009/D-012 in `ai-notes/decisions.md`) — enough for building and testing
  logic, not for training at the scale either method needs.
- **Nothing here is provably correct.** All of it is measured statistically
  on a benchmark, not proven mathematically, and only generalizes as far as
  it's actually been tested — held-out objects, layouts, paraphrases, and
  intervention types (Threats to validity,
  [`01-problem-statement-and-motivation.md`](01-problem-statement-and-motivation.md)).

## Primary deliverables

- A benchmark generator with reproducible, ground-truth world changes
- An oracle goal-feasibility and constraint checker
- Static, oracle, and adaptive policy baselines
- Self-supervised representation comparisons and probing results
- Multi-seed evaluation of feasibility prediction, goal achievement, intent
  violations, adaptation efficiency, and held-out-change generalization

## Document map

The numbered documents cover the problem definition, related work, architecture,
environment design, world-change taxonomy, policy design, training, evaluation,
roadmap, portfolio packaging, and experiment logging. The `ai-notes/` directory
tracks live decisions, risks, status, and work items. The rigorous equations,
assumptions, conditional properties, and code correspondence are collected in
[`math.md`](math.md).
