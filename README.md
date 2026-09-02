# Adaptive Task Recovery

A robot is part-way through a task when the world changes in a way it cannot
undo. An object is knocked out of reach. A goal gets blocked. Something drops
into the workspace that may or may not move again. The robot has to work out
which of its goals are still achievable, finish those, and leave alone what it
was told not to touch.

This repository contains a simulated benchmark for that problem, an audit that
checks whether such a benchmark measures what it claims to, and the full record
of what the audit found — including the parts that went against us.

<p align="center">
  <img src="media/demos/v4-router-montage.gif" width="900" alt="Three episodes of a Panda arm under different irreversible changes, each labelled with the recovery option the controller has committed to and how the episode resolved.">
</p>

<p align="center"><sub>
One controller, three mechanisms. It commits to a response almost immediately
when a cube is ejected, but observes for tens of steps before committing when a
goal is blocked — because a blockage that will clear and one that will not look
identical when they first appear. The outcome line separates them: the two
irreversible mechanisms end with one cube placed and one permanently gone,
while the reversible one ends with both placed.
</sub></p>

## The problem underneath the problem

Building a recovery benchmark is easy. Building one that actually measures
recovery is not, and the failure is quiet.

The usual design holds out a failure mechanism the model never trains on, then
asks whether it can compose an appropriate response. A large margin over a
learned baseline reads as evidence that it can. That inference is only valid if
the held-out mechanism cannot be identified by something lacking the capability
being tested — and checking that is not standard practice.

It should be. When we checked our own benchmark, a model seeing a single past
frame, with no memory and no sequence encoder, identified the held-out mechanism
as accurately as the recurrent model did.

## The control ladder

The audit scores a benchmark's held-out mechanism against four controls of
increasing capability, on identical inputs, splits and targets:

| Rung | Control | The question it answers |
|---|---|---|
| 1 | the current frame only | is the mechanism visible instantaneously? |
| 2 | one earlier frame | is it visible from a single past observation? |
| 2b | an order-free summary of the prefix | is it visible without temporal order? |
| 3 | a hand-written motion threshold | is it visible to a rule with no learning? |
| 4 | a recurrent model | does identifying it require temporal evidence at all? |

A lower rung **matches** rung 4 when the paired bootstrap interval on their
difference includes zero — resampling whole episodes for the simulated
benchmarks and object families for recorded trajectory data, pooling every
training seed. If a lower rung matches, the held-out mechanism is a shortcut,
and no composition claim follows from it however large the margin over a weak
baseline looked.

<p align="center">
  <img src="media/results/shortcut-ladder.png" width="960" alt="Held-out mechanism score by control rung across three benchmarks, and closed-loop recovery on permanent versus temporary obstruction for four arms.">
</p>

## What it found

Run on three benchmarks with an identical rung set:

| Benchmark | Best lower rung | Rung 4 | Difference | Verdict |
|---|---:|---:|---|:---:|
| `LearnedRecovery-v4` — ours, two-cube tabletop | 1.0000 | 1.0000 | +0.0000 [0.0000, 0.0000] | shortcut |
| `PegInsertionSide-v1` — contact-rich insertion | 0.0909 | 0.4015 | +0.3240 [0.1344, 0.5231] | none |
| Recorded real-robot trajectories, ten seeds | 0.7482 | 0.8108 | +0.0626 [0.0035, 0.1367] | none |

One positive, two negatives. The audit discriminates rather than firing
everywhere, which is what makes the positive worth reading.

That positive has a specific and slightly embarrassing cause. Our two ejection
directions were produced by *separate actors* — a forward pusher and a reverse
pusher — so identifying the mechanism reduced to noticing which one had moved.
A hand-written threshold reaches 97.40% closed-loop doing exactly that.

Worse, we had already fixed a shortcut here once. An earlier version leaked the
mechanism through instantaneous geometry, so we re-expressed every frame as a
displacement relative to the present. That worked: rung 1 falls to 0.0322. But
because every frame now carried a signed displacement, a single early frame
carried the whole answer instead. **Closing one leak opened a subtler one a rung
up**, and only a ladder of controls makes that visible.

### Which capability needs memory is not stable across benchmarks

On our benchmark, telling a permanent obstruction from a temporary one genuinely
requires temporal evidence. Both non-recurrent controls fail that pair in
*opposite* directions, each solving one side and scoring zero on the other,
while both recurrent models solve both.

On the contact-rich task it reverses. There the memoryless control is the
strongest arm on permanent blockage.

| Arm | ours: permanent | ours: temporary | contact-rich: permanent |
|---|---:|---:|---:|
| Recurrent, factorized | 0.9740 | 0.8438 | 0.4740 |
| Recurrent, unstructured | 0.9740 | 0.8420 | 0.5208 |
| One past frame | 0.0000 | 0.8438 | **0.5677** |
| Hand-written rule | 1.0000 | 0.0000 | 0.0000 |

So "temporal evidence is needed to judge persistence" is a property of the
benchmark it was measured on, not a fact about recovery.

### A benchmark can leak through its clock

The contact-rich task's features are expressed relative to the present, which
should leave a single-frame model with almost nothing. It nonetheless scored
0.80 and 0.84 on the two blockage conditions. The remaining input was elapsed
time.

Zeroing that one feature and changing nothing else:

| Model | Condition | With time | Without | Change |
|---|---|---:|---:|---:|
| Single frame | permanent blockage | 0.8034 | 0.6094 | **−0.1940** |
| Single frame | temporary blockage | 0.8424 | 0.6027 | **−0.2398** |
| Single frame | ejection | 0.6033 | 0.6071 | +0.0038 |
| Recurrent | permanent blockage | 0.7362 | 0.7468 | +0.0106 |

The effect lands on the two blockage conditions and nowhere else, and the
recurrent model is untouched. Mechanisms that end episodes at different times
produce different duration distributions, so an elapsed-time feature encodes
which mechanism fired. **Any recovery benchmark carrying one should ablate it.**

## What this does and does not establish

The audit is the contribution, and it holds: a reusable procedure with a
statistical criterion, validated by returning different verdicts on benchmarks
that look equivalent by construction.

The recovery architecture is reported as a **negative result**. On recorded
real-robot trajectories it is statistically indistinguishable from a
capacity-matched plain recurrent model (−0.0021, interval [−0.0123, +0.0069],
ten seeds). On the benchmark without a shortcut it reaches 0.0199 on genuinely
observed held-out prefixes.

Two preregistered gates were frozen before the runs they scored, and both
failed; both are kept as results rather than removed. A continuation intended to
strengthen the nominal controller failed all four checks and *degraded*
competence. A redesigned environment intended to remove the actor-identity
shortcut failed three of four physics checks — the idea was right, the
implementation did not produce the motion it was supposed to.

Full detail, including a verdict that changed three times as the audit was
corrected, is in
[`docs/30-recovery-audit-protocol.md`](docs/30-recovery-audit-protocol.md).

## Limitations

Stated here rather than buried, because they bound everything above.

The task is easy. Five-centimetre cubes onto nine-centimetre pads with a
four-centimetre tolerance and no orientation requirement; the same
pick-and-place primitive reaches 98.31% elsewhere in this repository. The
tolerance is loose enough to be visually ambiguous — a cube counted as placed
can overhang the edge of its pad.

Interventions are scripted exogenous events at a fixed step, so recovery here
means recognising a goal is gone and completing the rest. That is goal
filtering, not recovery from the robot's own execution failure. We tried to
replace them with emergent failures and found the nominal controller's own
failures to be 98.8% a single mode — not enough variety to route over.

All closed-loop control is simulated. The real-robot evidence is offline
inference on recorded trajectories.

Only the benchmark we built flags. The contact-rich task and the external
trajectories do not, so the evidence supports a claim about how a benchmark of
this construction leaks, not a claim about the class.

[`docs/19-evidence-standards.md`](docs/19-evidence-standards.md) states the
standard a result must meet here before it is called established.

## Repository layout

```
src/atr/envs/        recovery environments, including the contact-rich extension
src/atr/policies/    routers, matched controls, and the hand-written baseline
scripts/             the audit, training, evaluation and figure pipelines
docs/                protocol, evidence standards, and the design record
ai-notes/            decision log, risks, and the living status tracker
configs/             frozen experiment configurations and their gates
results/             committed measurements cited by the documents above
tests/               contract tests for the environments and controls
```

Entry points worth knowing:

- [`scripts/audit_shortcut_ladder.py`](scripts/audit_shortcut_ladder.py) — the
  control ladder and its statistical criterion.
- [`scripts/plot_shortcut_ladder.py`](scripts/plot_shortcut_ladder.py) — renders
  the figure above from committed measurements.
- [`src/atr/policies/option_router.py`](src/atr/policies/option_router.py) — the
  recurrent models and the matched controls they are compared against.
- [`src/atr/policies/heuristic_option_router.py`](src/atr/policies/heuristic_option_router.py)
  — the hand-written rule, written over the same inputs so the comparison is
  fair.

## Getting started

```bash
pip install -e .
pytest tests -q --ignore=tests/drafts --ignore=tests/spikes
```

That runs the maintained contract tests for the environments, the routers and
the audit — 126 tests, about a minute and a half. `tests/drafts/` and
`tests/spikes/` hold the per-candidate tests written alongside experiments that
were later abandoned. They are kept as part of the record but are not
maintained: some expect a GPU or a checkpoint that is not in the repository and
will stall rather than fail, so do not include them in a routine run.

Training and closed-loop evaluation expect a GPU, and the checked-in Slurm
wrappers show the exact invocations used to produce every committed result.

Experiment configurations are immutable once frozen: seed families, thresholds
and checkpoint digests are recorded before a run and never edited afterwards.
Reserved seed families stay unopened until the work they gate is complete.

## Naming

Several numbering schemes accumulated during development and occasionally reuse
the same token for different things. A counter is not a quality ordering — a
higher number is a later attempt, not a better one, and most were rejected.
[`docs/31-naming-and-identifier-key.md`](docs/31-naming-and-identifier-key.md)
maps every counter to what it refers to.

## License

See [LICENSE](LICENSE).
