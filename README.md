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

## Scope and status

The first version is simulation-only and targets a **simulated humanoid** carrying
out multi-goal, visually observable, object-centric tasks. The feasibility and
intent modules are embodiment-agnostic, while a humanoid control layer provides
navigation, reaching, grasping, and safe whole-body skills. A simpler embodiment
may be used for early research debugging, but humanoid evaluation is a required
project milestone.

The repository is still formally **Pre-Phase 0** — `src/atr/` doesn't exist yet,
and nothing has been promoted out of `spikes/` — but the spike work there
(`spikes/task_schema_draft/`) is substantial and tested: a goal-graph schema,
oracle feasibility, an intent guard, confirmed embodiment-agnostic across four
robot/scene combinations (Panda arm, a Unitree G1 humanoid, a real ReplicaCAD
apartment with a mobile Fetch robot, and G1 placed in that same apartment), plus
one working build-up-order pass through language parsing, zero-shot vision,
self-supervised representations, and a learned (Q-learning) policy — over 90
tests passing. None of it is committed architecture: the schema it's all built
on needs teammate review before promotion (see
[`ai-notes/review-request-task-schema.md`](ai-notes/review-request-task-schema.md)).
See [STATUS.md](STATUS.md) for current work and [docs/](docs/) for the study design.

## Roadmap

1. Select a humanoid-capable environment and define the goal/constraint task schema.
2. Build deterministic irreversible-world-change interventions.
3. Train and compare self-supervised visual representations.
4. Learn goal-feasibility prediction and calibrated uncertainty.
5. Train the adaptive vision-language RL policy and intent guard.
6. Evaluate compositional and out-of-distribution generalization.

See the [full roadmap](docs/11-roadmap-and-milestones.md).

## Two-person team

Both contributors jointly own the benchmark, schemas, integration contracts,
and final evaluation. After that shared foundation:

- **Person A — representation and feasibility:** visual/language model selection,
  self-supervised representations, goal graphs, change inference, per-goal
  feasibility, calibration, and abstention.
- **Person B — policy and humanoid execution:** simulator/asset integration,
  low-level skills, static and adaptive RL policies, intent guard, and policy baselines.

Integration is continuous: Person B first develops against oracle feasibility,
Person A develops against recorded trajectories, and both replace the oracle
with learned beliefs through a versioned shared interface.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
