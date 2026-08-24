# Adaptive Task Recovery

Adaptive Task Recovery (ATR) is a research project about **intent-preserving
adaptation after irreversible world changes**. It studies whether a
vision-language reinforcement learning agent can use self-supervised visual
representations to determine which language-specified goals remain feasible and
revise its strategy to achieve as much of the original intent as possible.

<p align="center">
  <img src="media/demos/fetch-real-pick-and-place.gif" width="330" alt="A Fetch robot navigates to a can, reaches down, closes its gripper, lifts and carries the can across a real ReplicaCAD apartment, and places it on a tray.">
  <img src="media/demos/fetch-safety-detour.gif" width="330" alt="The same Fetch robot screens its planned route, finds a protected object sitting on it, and replans a detour instead of pushing through.">
</p>

<p align="center"><sub>
Real ManiSkill3 simulation, real collision-aware path planning, real inverse
kinematics, a real contact-force-verified grasp — no scripted camera moves,
no teleportation. See <a href="#what-this-project-does-not-claim">what this
project does not claim</a> for the honest boundary of what "real" covers here.
</sub></p>

## Research question

A robot gets an instruction with more than one goal. Partway through, the
world changes in a way that can't be undone — something breaks, disappears,
or a path closes. Can the robot tell which goals are still possible, still
do whichever of them remain achievable, and never fake success by doing
something it was never asked to do?

> Can a vision-language reinforcement learning agent, equipped with
> self-supervised visual representations, learn to identify which
> language-specified goals remain feasible after unforeseen and irreversible
> world changes, and adapt its task strategy to maximize goal achievement
> without violating the original intent?

See [`docs/00-project-overview.md`](docs/00-project-overview.md) for the
full breakdown, including the build-up order this project actually follows
(language, then vision, then self-supervised representations, then a
learned policy — one capability at a time, each checked before the next).

Examples include an object becoming unavailable, a container breaking, a route
becoming blocked, or a limited resource being consumed. These changes make the
original plan invalid and may make some goals impossible. The challenge is to
distinguish infeasibility from temporary difficulty, reason over multiple goals
and constraints, and choose an acceptable partial or alternative completion.

## Results

Five hypotheses, each with real evidence — paired seeds, bootstrap confidence
intervals, and real ManiSkill3 simulator episodes, not toy numbers. Full
detail and every underlying number is in
[`ai-notes/decisions.md`](ai-notes/decisions.md) (D-001–D-123); this is the
short version.

| | Hypothesis | Result |
|---|---|---|
| **H1** | A perceptual feasibility signal (not just privileged simulator state) is usable | **Confirmed.** Real zero-shot CLIP perception matches oracle behavior exactly on the project's own success-criteria benchmark — same recall, real reduction in wasted steps. A robustness gap was found by actually running the benchmark, then fixed and re-verified, not assumed away. |
| **H2** | Feasibility-aware policies beat blindly continuing the plan | **Confirmed.** 30-seed paired benchmark on the real Fetch/apartment stack: identical goal recall, a statistically real drop in wasted steps (bootstrap CI excludes zero). A policy trained on two apartment layouts matched the oracle exactly on a third, never-seen layout, 10/10 seeds. |
| **H3** | The safety guard does real work, not "safe by doing nothing" | **Confirmed.** Built the two adversarial cases meant to break this claim — a goal that directly conflicts with a protected object, and a case where the guard turned out to be *too permissive*. Both found real bugs, both fixed, both locked into regression tests. Extended to real collision-aware navigation with live detours and fail-closed stops, verified under seed variance, not one staged scenario. |
| **H4** | A factorized language representation generalizes; memorization doesn't | **Confirmed.** The real parser hits 100% across train, held-out paraphrase, and held-out composition splits. A hand-built monolithic memorizer hits 100% on train and 0% on anything held out. Re-verified on the full 180-case combinatorial sweep of the object pool, not a hand-picked sample. |
| **H5** | Calibrated abstention beats forced decisions | **Confirmed — conditionally.** Abstaining only wins when a wrong forced decision is the expensive mistake; tested both directions honestly, not just the flattering one, and confirmed with 30-seed bootstrap intervals that exclude zero in both directions (`[−0.20, −0.06]` where abstention wins, `[+0.10, +0.14]` where it loses). The unconditional version of this hypothesis was wrong; the conditional one is real. |

**How this held together:** every comparison is paired-seed and bootstrapped
(docs/10's predeclared protocol, used consistently); every required baseline
(domain-randomized, symbolic replanner, imitation learning, frame-difference
detector) is actually built and compared, not asserted as future work; the
full test suite (300+ tests, real simulator episodes, no mocks on load-bearing
paths) runs before anything is called done, and has caught real regressions
targeted tests missed; and negative results are kept, not edited out — a CLIP
calibration that didn't transfer, a hypothesis that only holds conditionally,
a config value that never did what its own comment claimed, all reported as
found, with the fix or the disclosed gap next to it.

**What's still genuinely open:** collision geometry is still spheres and
points, not full robot/object meshes; CLIP calibration is scene-specific and
doesn't transfer automatically (confirmed by testing it on a new apartment
layout and watching it fail); everything runs in simulation on CPU, no real
robot and no large-scale learned visual policy; and one object's placement
choice in the two original apartment layouts still has no explanation anyone's
found. None of these are blocking — they're disclosed scope, not surprises.

### What this project does not claim

Both demos above are real, not scripted: real navigation, real collision-aware
replanning, real inverse kinematics, and a real contact-force-verified grasp
(`agent.is_grasping()`, ManiSkill3's own detector — checked at every stage,
not assumed). The pick-and-place demo is built in a separate, additive module
(`src/atr/envs/tidy_up_replicacad_manipulation.py`, D-124), deliberately kept
apart from the `attempt_goal()` navigate-then-teleport contract every H1-H5
result and every navigation-safety decision (D-091–D-123) is built on across
300+ tests — that contract stays exactly as it was; changing it project-wide
for a demo's visual benefit would risk the whole evidence base for no
research reason. So concretely:

- **What's real:** Fetch, one object (the potted meat can), one scene —
  navigate, reach, grasp, lift, carry across the apartment while still
  gripping, place, release, and a real physics-settled final position,
  verified with the project's own `goal_achieved()` check, not a custom one.
- **What isn't (yet):** this hasn't been benchmarked across seeds or objects,
  isn't wired into any policy the H1–H5 results depend on, and doesn't cover
  the humanoid embodiment — a real analytic-Jacobian IK check there
  (`src/atr/control/ik_solver.py`) found, and kept as a disclosed regression
  test rather than hidden, that neither goal object is within true contact
  range from G1's calibrated standing position. That's a measured kinematic
  limit of that setup, not a missing feature, and it's why this capability
  was built for Fetch specifically rather than assumed to transfer.

## Planned system

- A language parser that represents goals, priorities, and hard constraints
- A self-supervised visual encoder that learns object and scene representations
- A world-change and goal-feasibility estimator
- A language-conditioned RL policy that replans or changes strategy
- An intent guard that rejects actions that contradict the instruction
- A benchmark with controlled irreversible changes and held-out changes
- Evaluation against static-policy, replanning, and representation baselines

See [`docs/03-system-architecture.md`](docs/03-system-architecture.md) for
the module-boundary and ownership diagram behind the list above.

## Scope and status

The first version is simulation-only and targets a **simulated humanoid** carrying
out multi-goal, visually observable, object-centric tasks. The feasibility and
intent modules are embodiment-agnostic, while a humanoid control layer provides
navigation, reaching, grasping, and safe whole-body skills. A simpler embodiment
may be used for early research debugging, but humanoid evaluation is a required
project milestone.

The core task schema — `Goal`/`Constraint`/`GoalGraph`, oracle feasibility,
the intent guard — is accepted and promoted to [`src/atr/`](src/atr/) (D-037).
**Read the review's status banner before treating that as settled**: it was
self-resolved by the project owner, not independently reviewed by the
teammate it was written for — see
[`ai-notes/review-request-task-schema.md`](ai-notes/review-request-task-schema.md).
The promotion sweep is effectively complete: language parsing, zero-shot
CLIP feasibility, every required baseline (static/oracle-feasibility/naive-
substitution/domain-randomized/symbolic-replanner/imitation-learning),
Q-learning, every embodiment/scene environment (a Panda arm, a Unitree G1
humanoid, a real ReplicaCAD apartment with a mobile Fetch robot including a
full collision-aware navigation-safety stack, and G1 placed in that same
apartment across three independently verified scene layouts), the end-to-end
pipeline, the evaluation harness, a queryable dataset-split registry
(instruction-, intervention-, and scene-layout-level), a log interface, and
experiment tracking all live in `src/atr/`, tested and `git`-committed
architecture. Only `dinov2_probe.py` remains spike-stage in
`spikes/task_schema_draft/` — DINOv2 wired into a real live decision loop,
a genuine robustness gap found and closed (D-054/D-055), still not
promotion-ready. 300+ tests passing.
See [STATUS.md](STATUS.md) for current work, [`ai-notes/decisions.md`](ai-notes/decisions.md)
for the full decision-by-decision record, and [docs/](docs/) for the study design.

## Roadmap

1. Select a humanoid-capable environment and define the goal/constraint task schema.
2. Build deterministic irreversible-world-change interventions.
3. Train and compare self-supervised visual representations.
4. Learn goal-feasibility prediction and calibrated uncertainty.
5. Train the adaptive vision-language RL policy and intent guard.
6. Evaluate compositional and out-of-distribution generalization.

See the [full roadmap](docs/11-roadmap-and-milestones.md).

## Scope areas

The benchmark, schemas, integration contracts, and final evaluation are
shared foundations. Two work areas sit on top of that foundation:

- **Representation and feasibility:** visual/language model selection,
  self-supervised representations, goal graphs, change inference, per-goal
  feasibility, calibration, and abstention.
- **Policy and humanoid execution:** simulator/asset integration,
  low-level skills, static and adaptive RL policies, intent guard, and policy baselines.

Integration is continuous: policy work develops against oracle feasibility,
representation work develops against recorded trajectories, and both
replace the oracle with learned beliefs through a versioned shared interface.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
