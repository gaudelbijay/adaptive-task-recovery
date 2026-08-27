---
title: References and Reading Queue
status: verified-core
last_updated: 2026-08-27
---

# References and Reading Queue

Core entries below were checked against the paper or official project source.
The remaining sections are a reading queue and must not be cited as completed
coverage.

## Verified core bibliography

- Ahn, M., et al. (2022). *Do As I Can, Not As I Say: Grounding
  Language in Robotic Affordances*. arXiv:2204.01691; later CoRL 2022.
  https://arxiv.org/abs/2204.01691
- Huang, W., et al. (2022). *Inner Monologue: Embodied Reasoning through
  Planning with Language Models*. arXiv:2207.05608; later CoRL 2022.
  https://arxiv.org/abs/2207.05608
- Ren, A. Z., et al. (2023). *Robots That Ask For Help: Uncertainty Alignment
  for Large Language Model Planners*. CoRL 2023. arXiv:2307.01928.
  https://arxiv.org/abs/2307.01928
- Driess, D., et al. (2023). *PaLM-E: An Embodied Multimodal Language Model*.
  arXiv:2303.03378. https://arxiv.org/abs/2303.03378
- Brohan, A., et al. (2023). *RT-2: Vision-Language-Action Models Transfer Web
  Knowledge to Robotic Control*. arXiv:2307.15818.
  https://arxiv.org/abs/2307.15818
- Liang, J., Huang, W., Xia, F., Xu, P., Hausman, K., Ichter, B., Florence, P.,
  and Zeng, A. (2023). *Code as Policies: Language Model Programs for Embodied
  Control*. arXiv:2209.07753. https://arxiv.org/abs/2209.07753
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y.
  (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR
  2023. arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., and
  Topcu, U. (2018). *Safe Reinforcement Learning via Shielding*. AAAI 2018.
  arXiv:1708.08611. https://arxiv.org/abs/1708.08611
- Carr, S., Jansen, N., Junges, S., and Topcu, U. (2022). *Safe Reinforcement
  Learning via Shielding under Partial Observability*. arXiv:2204.00755.
  https://arxiv.org/abs/2204.00755
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017).
  *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
  https://arxiv.org/abs/1707.06347
- Tao, S., et al. *ManiSkill3: GPU Parallelized Robotics Simulation and
  Rendering for Generalizable Embodied AI*. arXiv:2410.00425 (2024).
  https://arxiv.org/abs/2410.00425. Official implementation and PPO task
  commands: https://github.com/mani-skill/ManiSkill and
  https://github.com/mani-skill/ManiSkill/blob/v3.0.0b22/examples/baselines/ppo/baselines.sh.
  The authors' baseline learning-curve report is
  https://wandb.ai/stonet2000/ManiSkill/reports/PPO-Results--VmlldzoxMDQzNDMzOA

## Comparison notes

SayCan supplies the feasibility-grounded skill-selection comparison; Inner
Monologue supplies the feedback-driven replanning comparison; KnowNo supplies
the calibrated-abstention comparison; shielding supplies the hard-constraint
comparison; and PPO/ManiSkill supply the real-control learning comparison.
Because their embodiments, observations, goals, and success definitions differ,
published success percentages must not be placed beside ATR numbers as if they
were measured on a shared test set.

## Vision-language embodied control

- Vision-language-action models and language-conditioned robotic policies
- Embodied instruction-following benchmarks
- Language-conditioned task-and-motion planning
- Compositional language grounding and referring expressions

## Self-supervised visual learning

- Contrastive image representation learning
- Masked image modeling
- Self-distillation without labels
- Temporal contrastive and video prediction objectives
- Object-centric representation learning and scene graphs
- Affordance-aware visual representation learning

## Reinforcement learning and planning

- Goal-conditioned and multi-goal reinforcement learning
- Hierarchical RL and options
- Successor features and generalized policy improvement
- Model-based RL and latent world models
- Nonstationary, hidden-parameter, and change-point MDPs
- Constrained MDPs and safe exploration

## Humanoid execution layer

- Humanoid loco-manipulation and whole-body control
- Reusable navigation, reach, grasp, and place skills
- Vision-language-action policies evaluated on humanoid platforms
- Safety-constrained balance and collision avoidance
- Skill-success and manipulation-reachability estimation

## Feasibility and adaptation

- Reachability and value-based feasibility estimation
- Learned precondition/effect and affordance models
- Continual test-time adaptation without catastrophic forgetting
- Plan repair and execution monitoring after exogenous events
- Resource-constrained and oversubscription planning

## Intent constraints

- Runtime shielding and action masking
- Reward machines, temporal logic, and specification-guided RL
- Constraint inference from language
- Reward hacking and specification gaming evaluations
- Selective prediction, uncertainty calibration, and abstention

## Benchmark methodology

- Causal and counterfactual benchmark design
- Dataset leakage and shortcut-learning audits
- Compositional and out-of-distribution generalization
- Paired-seed evaluation, bootstrap intervals, and RL reproducibility

## Reading-note fields

For each source record: setting, observations, language source, change model,
feasibility definition, adaptation mechanism, constraint mechanism, baselines,
generalization split, limitations, and relevance to an ATR hypothesis.
