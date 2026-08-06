# Adaptive Task Recovery

Adaptive Task Recovery (ATR) is a research project about **intent-preserving
adaptation after irreversible world changes**. It studies whether a
vision-language reinforcement learning agent can use self-supervised visual
representations to determine which language-specified goals remain feasible and
revise its strategy to achieve as much of the original intent as possible.

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

The repository is in **Phase 0**. The core task schema — `Goal`/`Constraint`/
`GoalGraph`, oracle feasibility, the intent guard — is accepted and promoted
to [`src/atr/`](src/atr/) (D-037). **Read the review's status banner before
treating that as settled**: it was self-resolved by the project owner, not
independently reviewed by the teammate it was written for — see
[`ai-notes/review-request-task-schema.md`](ai-notes/review-request-task-schema.md).
The promotion sweep is now effectively complete: language parsing, zero-shot
CLIP feasibility, the policy baselines (static/oracle-feasibility/naive-
substitution), Q-learning, imitation learning, every embodiment/scene
environment (Panda arm, a Unitree G1 humanoid, a real ReplicaCAD apartment
with a mobile Fetch robot, and G1 placed in that same apartment), the
end-to-end pipeline, the evaluation harness, a queryable dataset-split
registry (instruction- and intervention-level), a log interface, and
experiment tracking all live in `src/atr/`, tested and `git`-committed
architecture. Only `dinov2_probe.py` remains spike-stage in
`spikes/task_schema_draft/` — DINOv2 wired into a real live decision loop,
a genuine robustness gap found and closed (D-054/D-055), still not
promotion-ready. 154 tests passing.
See [STATUS.md](STATUS.md) for current work and [docs/](docs/) for the study design.

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
