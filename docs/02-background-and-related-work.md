---
title: Background and Related Work
status: verified-core
last_updated: 2026-08-28
---

# Background and Related Work

This document defines the literature map used for the current experiments.
Exact citations and primary-source links are in [references.md](references.md).
Numerical results from different robot embodiments and task suites are not
treated as a common leaderboard.

## Closest verified systems

- **SayCan** ranks pretrained skills using language-model relevance and
  value-function affordance grounding. It is the closest reference for
  selecting feasible high-level robot skills, but does not benchmark recovery
  after hidden persistent changes that irreversibly remove goals.
- **Inner Monologue** closes the planning loop using success, scene, and human
  feedback. It motivates replanning after execution feedback, but does not
  expose ATR's per-goal feasibility, resource-contention, and explicit
  side-effect-constraint evaluation.
- **KnowNo** calibrates when an LLM planner should ask for help using conformal
  prediction. It is the closest reference for abstention under uncertainty;
  ATR instead measures autonomous action selection after world changes and the
  resulting goal/safety tradeoff.
- **Shielding** filters or corrects actions against formal safety properties.
  It motivates ATR's intent guard. Our effect-aware audit additionally shows
  why checking only a skill's named target is insufficient when its physical
  effects can violate another object's constraint.
- **PPO and ManiSkill** provide the low-level continuous-control algorithm and
  task-specific reference hyperparameters for the non-teleport manipulation
  experiments. They are execution baselines, not recovery policies.
- **MoPA-PD visual policy distillation** is the closest methodological
  precedent for ATR's state-teacher-to-visual-controller bootstrap: it combines
  visual behavioral cloning from a state-dependent planner-augmented policy
  with subsequent vision-based reinforcement learning. ATR's DAgger variants
  add on-policy student coverage and test persistent goal removal, but remain
  privileged-training methods rather than pure pixel-only RL.
- **Sequential Dexterity** learns transition feasibility for chaining
  manipulation subpolicies, including switching to recover from failures and
  bypassing redundant stages. This is especially relevant to ATR's ordered
  suffix behavior. Its unit of control is a chain of dexterous subpolicies;
  ATR instead evaluates one continuous goal-order-conditioned policy after an
  exogenous stage becomes physically impossible.
- **SPIRE** combines task-and-motion-planning decomposition, imitation, and RL
  for long-horizon contact-rich manipulation. It is a strong precedent for
  using structured expert guidance rather than expecting pixel RL to discover
  long sequences from scratch. Its published task suites do not isolate
  unannounced irreversible goal loss or protected-object violations.
- **DEMO³** combines demonstrations, learned stage-wise dense rewards, and a
  visual latent world model for long-horizon manipulation. It is the closest
  recent precedent for using a small amount of expert data to make multi-stage
  pixel control tractable. ATR's intervention changes which requested stage is
  feasible during execution, whereas DEMO³ assumes the demonstrated stage
  structure remains achievable; its published percentages are therefore not a
  shared-benchmark baseline.
- **MSDP** self-supervises a multimodal representation with masked
  reconstruction and then uses asymmetric actor--critic training for
  contact-rich manipulation. It strengthens the motivation for ATR's
  restricted actor plus privileged training-only critic/auxiliary targets,
  while differing in modality (vision, force, and proprioception), pretraining
  protocol, and task suite.
- **MENTOR** is a recent visual-RL mixture-of-experts method that combines
  task-oriented perturbations with expert specialization and reports both
  simulated and real-robot manipulation results. It is an important
  architectural comparator for ATR's dual-specialist student, but MENTOR does
  not evaluate an unannounced irreversible goal-removal event or ATR's
  protected-object constraint. Its published percentages are therefore not a
  head-to-head baseline.
- **Maniwhere** combines multi-view representation learning, a spatial
  transformer, and curriculum randomization for visual generalization across
  manipulation tasks and robot platforms. It motivates a future camera/domain
  generalization axis; ATR's present frozen experiment instead isolates
  persistent task-feasibility change under one declared visual protocol.
- **Masked-modality training for sensor failure** randomly removes visual or
  proprioceptive inputs during RL and evaluates robustness to missing sensors.
  This is a useful perturbation-training comparator, but sensor loss changes
  observability while ATR's intervention changes which requested world goal is
  physically achievable. The two failure models should not be conflated.
- **PaLM-E, RT-2, and Code as Policies** broaden embodied multimodal reasoning,
  end-to-end vision-language-action control, and language-generated robot
  programs respectively. They are important representation/execution
  comparators, but their published evaluations do not isolate hidden persistent
  changes, recomputed remaining-goal feasibility, and side-effect guards.
- **ReAct** interleaves reasoning with environment actions in interactive text
  domains. It supports the closed-loop motivation but is not a physical
  manipulation or formal intent-constraint baseline.
- **RecoveryChaining** learns a local recovery policy that returns a failed
  manipulation system to a state where nominal model-based controllers can
  resume. It is the closest learned-control recovery comparator. ATR's
  integrated benchmark instead makes an exogenous change permanently remove a
  language goal, so correct behavior may be to abandon that goal and complete
  the feasible suffix rather than restore the nominal trajectory.
- **REFLECT** turns multimodal execution histories into failure explanations
  that guide corrective high-level plans. It assumes the rest of the
  environment remains static and identifies low-level control failure as a
  limitation; ATR isolates persistent external changes and trains the
  continuous policy directly, but does not yet provide REFLECT's open-ended
  explanation interface.
- **Autonomous Interactive Correction** uses interaction feedback to correct
  low-level SE(3) contact-pose predictions for articulated objects. It is a
  corrective-control comparator, while ATR evaluates ordered multi-goal
  completion after a goal becomes infeasible and a protected object must not
  move.
- **Failure-Aware RL / FailureBench** combines a safety critic with an offline
  recovery policy to reduce intervention-requiring failures during real-world
  online RL. Its safety/recovery framing is directly relevant; ATR's current
  evidence is simulation-only and instead reports an explicit hard-constraint
  violation rate and safety-qualified task success under a controlled
  irreversible intervention.

## Vision-language embodied agents

Review language-conditioned control, vision-language-action models, embodied
instruction following, and task-and-motion planning with language. The relevant
gap is not instruction following in a static world, but recomputing the feasible
goal set after an unannounced persistent change.

## Self-supervised visual representation learning

Compare contrastive, masked-image-modeling, self-distillation, and temporal or
object-centric objectives. The useful representation must encode object state,
relations, affordances, and change—not merely image similarity. Frozen,
fine-tuned, and task-reward-only encoders are necessary baselines.

For online visual control, DrQ-v2 and MENTOR are model-free algorithmic
references; Maniwhere is a visual-generalization reference; CP3ER is a
consistency/diffusion-policy stability reference; and DreamerV3, Dreamer 4,
TD-MPC2, and DEMO³ are world-model references. Dreamer 4 is the newest numbered
Dreamer found in the primary literature as of 2026-08-28, but its demonstrated
RL control domain is offline Minecraft rather than robot manipulation. Its
robotics result is world-model interaction prediction, not a manipulation
policy result. NE-Dreamer is the repository's implemented decoder-free
temporal-prediction pilot.

CP3ER is especially relevant to the disclosed V22 failure: it reports visual
policy degradation under actor--critic training and stabilizes a consistency-
model policy using sample-based entropy and prioritized proximal experience
regularization. It is nevertheless not the same intervention as ATR's failed
Gaussian augmentation KL: CP3ER changes the policy class and is evaluated with
off-policy Q-learning on DeepMind Control and Meta-World, whereas V22 adds a
separate invariance loss to on-policy PPO. V24's bounded action-consistency
pilot addresses V22's numerical failure only and is not described as CP3ER.

None of these systems supplies a directly comparable published score for ATR's
custom ordered-removal benchmark, so they define algorithm families and
ablations—not an external numerical leaderboard. The final experiment
therefore compares restricted RGB PPO, asymmetric training, temporal SSL,
DAgger, representation regularization, and privileged progress prediction
under identical V3 dynamics, seeds, budgets, and held-out resets.

## Goal-conditioned and hierarchical RL

Goal-conditioned RL, successor features, hierarchical policies, and constrained
MDPs provide tools for multi-goal optimization and policy reuse. ATR differs by
making language-defined goal feasibility nonstationary and partially observed.

## Humanoid loco-manipulation and skill interfaces

Review reusable humanoid navigation, reaching, grasping, whole-body control, and
vision-language-action skill interfaces. These are enabling components rather
than the primary novelty. The study must distinguish controller reachability
from goal infeasibility and document the embodiment assumptions behind its oracle.

## Continual adaptation and nonstationary MDPs

Nonstationary and hidden-parameter MDP work addresses changing dynamics or task
conditions. ATR focuses on discrete, persistent world-state changes that alter
the reachable goal set, often within a single episode and without online weight
updates.

## Affordance and feasibility prediction

Affordance learning, reachability/value estimation, and precondition-effect
models motivate the feasibility head. “Feasible” must be defined relative to the
current state, remaining horizon/resources, available actions, and constraints.

## Constrained decision-making and intent preservation

Constrained RL, shielding, runtime verification, reward machines, and temporal
logic offer operational ways to prevent hard violations. The project should use
machine-checkable constraints while testing whether language encodings preserve
their link to those constraints.

## Closest-baseline matrix

| Capability | SayCan | Inner Monologue | KnowNo | Shielding | ATR benchmark |
|---|---:|---:|---:|---:|---:|
| Grounds high-level robot skills | Yes | Yes | Yes | No | Yes |
| Uses closed-loop environment feedback | Limited | Yes | Optional | Yes | Yes |
| Calibrated abstention/help | No | Human feedback | Yes | No | Conservative baseline |
| Persistent hidden change changes goal feasibility | Not isolated | Not isolated | Ambiguity focus | Safety focus | Yes |
| Reports remaining-goal efficiency after change | No | No | No | No | Yes |
| Checks explicit side-effect constraints | No | No | No | Formal action safety | Yes |
| Includes non-teleport continuous manipulation control | Physical skills | Physical skills | Physical skills | Domain-dependent | Separate PPO track |

This table is a task-definition comparison, not a claim that ATR outperforms
these systems on their published benchmarks. The quantitative comparisons in
this repository use identical ATR cases, paired seeds, and common evaluators.

## Recovery-system comparison

| System | Recovery unit | Failure/change | Low-level control | Explicit protected-object metric |
|---|---|---|---:|---:|
| Inner Monologue | LLM replanning | Execution feedback | Pretrained skills | No |
| REFLECT | Explanation + replan | Execution/planning failure | Existing skills | No |
| RecoveryChaining | Learned local policy + nominal options | Nominal-controller failure | Yes | No |
| Autonomous Interactive Correction | Corrected contact pose | Articulated interaction failure | Pose execution | No |
| Failure-Aware RL | Safety critic + recovery policy | Intervention-requiring failure | Yes | No (IR-failure metric) |
| Sequential Dexterity | Transition feasibility + chained subpolicies | Failed/redundant stage | Yes | No |
| ATR dual-specialist RGB controller (visual series V19) | One restricted-input PPO policy conditioned on a parsed goal-order encoding | Persistent exogenous goal loss | Yes; continuous joint control | Yes; 1.30% strict / 3.65% nominal violations |

The ATR row now describes the completed three-seed screen: 96.35% strict safe,
91.41% nominal safe, and 97.06%/95.69% safe success on the two physical-removal
branches (768 episodes per regime). It uses privileged teachers, labels, and a
state critic during training and is simulation-only. No cited system evaluates
the same ordered irreversible-removal protocol, so these values are not placed
beside published percentages from different embodiments or benchmarks. The
fixed five-seed confirmation remains in progress.

## Literature-review protocol

Record the task setting, observation modality, source of language, change model,
feasibility definition, adaptation mechanism, constraint mechanism, and
generalization split for every study. This prevents grouping superficially
similar systems that answer different questions.
