# Adaptive Task Recovery

Adaptive Task Recovery (ATR) is a learning-based framework for helping humanoid
robots detect task failures and recover autonomously. The project focuses on
failures such as slipped objects, missed grasps, unexpected contact, sensor
dropout, and loss of balance.

The system is designed to:

1. Detect when execution has departed from the expected task behavior.
2. Select an appropriate recovery strategy.
3. Return the robot to a state where it can resume the original task, or abort
   safely when recovery is not possible.

## Project scope

The project is simulation-only and targets tabletop manipulation and basic
standing-balance recovery. Policies will be trained and evaluated in
[ManiSkill](https://www.maniskill.ai/) using a simulated humanoid model.

Planned components include:

- Task policies for nominal behavior
- Failure injection for repeatable experiments
- Learned and threshold-based failure detectors
- A library of recovery policies
- A policy arbiter for task, recovery, and abort decisions
- Evaluation against no-recovery and scripted-recovery baselines

## Current status

This project is in the planning and setup stage. The research questions,
architecture, training plan, evaluation strategy, and roadmap are documented in
the [`ai-notes`](ai-notes/) directory.

## Roadmap

The main development phases are:

1. Set up ManiSkill and import a humanoid model.
2. Train baseline task policies.
3. Implement controlled failure injection.
4. Train and evaluate failure detectors.
5. Build recovery policies and the end-to-end system.
6. Package the benchmark, results, and demonstrations.

See the [full roadmap](ai-notes/11-roadmap-and-milestones.md) for details.

## License

This project is licensed under the terms in [LICENSE](LICENSE).
