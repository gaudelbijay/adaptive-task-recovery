---
title: Evidence Standards and Known Limitations
status: active
last_updated: 2026-09-02
---

# Evidence standards and known limitations

This document states what the repository's results support, what they do not,
and the standard a result must meet before it is described as established. It
is deliberately unflattering: every limitation here was found by testing a
claim this project had already made.

## The standard

A result is described as established only when all of the following hold.

1. Its comparison is input-matched. Every arm receives the same observations,
   the same frozen specialists, the same evaluation seeds, and the same
   execution settings. An arm that receives a mechanism the others cannot run
   is reported separately, never folded into the headline.
2. Every declared comparator has actually been run. A method list that names
   five arms and scores three does not support a claim about the strongest
   baseline.
3. Its uncertainty comes from a paired interval over the correlated unit —
   whole episodes for the simulated benchmarks, object families for external
   trajectory data — and pools every training seed.
4. Its threshold was fixed before the outcome was observed, and the outcome is
   reported against that threshold whether it passes or fails.
5. Any held-out mechanism has survived the capability ladder in
   [`30-recovery-audit-protocol.md`](30-recovery-audit-protocol.md). A
   mechanism a weaker control identifies as well as the model under test is a
   shortcut, and no composition claim follows from it.

## What the results support

The capability ladder detects when a benchmark's held-out mechanism is
identifiable without the capability the benchmark claims to test. Run on three
benchmarks with an identical rung set, it reports a shortcut on one and none on
the other two, so it discriminates rather than firing everywhere.

Four independent shortcuts are documented, each with a specific mechanism:
separate actors per ejection direction; a one-frame shortcut created by the
current-centering transform that removed an earlier one; a verdict that depends
on which non-recurrent control is nominated; and condition identity leaking
through an episode-duration feature.

## What the results do not support

**No general claim about recovery benchmarks.** The only benchmark the audit
flags is the simple one built here. The contact-rich task and the external
real-robot trajectories do not flag. The evidence supports a narrower claim
about how a benchmark of this construction can leak, not a claim about the
class.

**No claim that temporal evidence is required for persistence.** On the simple
benchmark, distinguishing a permanent obstruction from a temporary one requires
memory: both non-recurrent controls fail the pair in opposite directions. That
reverses on the contact-rich task, where the memoryless control is the strongest
arm on permanent blockage. The finding is scoped to the benchmark it was
measured on.

**No support for the factorized architecture.** On external trajectory data it
is statistically indistinguishable from a capacity-matched plain recurrent
model, at −0.0021 with a 95% interval of [−0.0123, +0.0069] over ten optimizer
seeds. On the benchmark without a shortcut it reaches 0.0199 on genuinely
observed held-out prefixes. The audit is the contribution; the architecture is
reported as a negative result.

**No real-robot closed-loop result.** All closed-loop control is simulated. The
real-robot evidence is offline inference on recorded trajectories.

## Limitations of the benchmark itself

**The task is easy.** Five-centimetre cubes onto nine-centimetre pads with a
four-centimetre tolerance and no orientation requirement. The same pick-and-place
primitive reaches 98.31% elsewhere in this repository. The additional difficulty
comes from the ordering constraint, the protected-object tolerance and the step
budget, not from the manipulation.

**The placement tolerance is loose enough to be visually ambiguous.** A cube
counted as placed can be overhanging the edge of its pad; measured distances at
one resolution step were 0.035 and 0.029 metres against a 0.04 limit.

**Interventions are scripted.** They are exogenous events fired at a fixed step
with known mechanisms. Recovery here means recognising that a goal is gone and
completing the remainder, which is goal filtering rather than recovery from
execution failure. Replacing them with emergent failures was attempted: the
nominal controller's own failures are 98.8% a single mode, so that route needs a
change of task geometry rather than more training.

## What would raise the standard

Closed-loop control on physical hardware, or failures that emerge from the
policy's own execution rather than from a scripted intervention. Neither is
reachable from the current setup without substantial new work, and two attempts
at intermediate steps were rejected by their own preregistered gates. Those
rejections are recorded in
[`30-recovery-audit-protocol.md`](30-recovery-audit-protocol.md) rather than
removed.
