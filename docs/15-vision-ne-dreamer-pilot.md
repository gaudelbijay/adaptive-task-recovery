# Vision NE-Dreamer pilot (August 27, 2026)

## Decision and claim boundary

The project uses the official NE-Dreamer implementation at commit
`7ca2193657ba42fb33e4c3e308c538d83dd393ac`. NE-Dreamer is a 2026,
decoder-free Dreamer variant that predicts future encoder embeddings. It was
selected over Dreamer4 because no author-maintained Dreamer4 training code was
available for this manipulation setting, and over DreamerV3 because
NE-Dreamer directly tests the requested self-supervised representation
hypothesis.

This pilot establishes a working, checkpointed pixel-based model-based RL
pipeline. It does **not** establish competitive manipulation performance.

## Algorithm audit (updated August 28, 2026)

There is no published **DreamerV5** method to use as a truthful algorithm name.
[Dreamer 4](https://arxiv.org/abs/2509.24527) is the newest numbered Dreamer
generation located in the primary literature. Its reported agent is trained
inside a scalable video world model for offline Minecraft control; that work
does not provide an author-maintained drop-in continuous-manipulation training
pipeline for this benchmark. [DreamerV3](https://arxiv.org/abs/2301.04104)
remains the broadly validated general online world-model baseline, while
[TD-MPC2](https://arxiv.org/abs/2310.16828) and
[DrQ-v2](https://arxiv.org/abs/2107.09645) are relevant latent-model and
model-free visual-control baselines respectively.

The selected [NE-Dreamer](https://arxiv.org/abs/2603.02765) implementation is
newer than those online-control baselines and directly targets temporal
self-supervised representations. Its negative result here therefore remains a
meaningful world-model baseline, but the 250k-step/ratio-32 pilot is not a fair
claim that DreamerV3, Dreamer 4, or TD-MPC2 cannot solve the task. A larger
world-model comparison is justified only after the controller curriculum
produces visual competence; otherwise it repeats an already diagnosed sparse
long-horizon control failure at much higher compute cost.

## Observation and control protocol

The policy receives only:

- one 64 by 64 RGB base-camera image;
- Panda joint position and velocity; and
- the two-token red-first/blue-first instruction.

Object poses, goal poses, sweeper poses, TCP pose, task progress, and oracle
unavailability are excluded from the policy observation. Simulator state is
used only to compute reward and evaluation labels. Every environment step uses
the Panda's continuous `pd_joint_delta_pos` controller. There is no teleport
control.

## Correctness issue found before the final pilot

The pinned upstream `bounded_normal` returned an unconstrained independent
Gaussian. Dreamer called `rsample()`, producing commands observed between
approximately -4 and +4. ManiSkill clipped those commands to [-1, 1], while
the replay buffer stored the original unclipped actions. The world model was
therefore conditioned on a command different from the one executed.

Job `1139094` was cancelled and excluded from results. The integration now
uses an elementwise straight-through bound for both environment interaction
and imagined rollouts, and the environment raises an error if an unbounded
command reaches the controller. Smoke job `1139164` verified action minima and
maxima of exactly -1 and +1 and verified checkpoint/replay resume.

## Corrected pilot

- Training Slurm array: `1139165`
- Final evaluation Slurm array: `1139185`
- Seeds: 9351, 4796, 1788
- Budget: 250,000 environment steps and 30,625 optimizer updates per seed
- Model: 15,074,831 optimized parameters (`size12M` upstream preset plus the
  NE-Dreamer projector and temporal transformer)
- Final evaluation: 256 deterministic episodes per seed with intervention
  probability 1.0

| Seed | Final return | Successes / episodes | First NE loss | Final NE loss |
|---:|---:|---:|---:|---:|
| 9351 | 1.176 | 0 / 256 | 839.339 | 143.158 |
| 4796 | 0.825 | 0 / 256 | 789.300 | 131.124 |
| 1788 | 0.296 | 0 / 256 | 830.447 | 139.721 |
| Aggregate | 0.766 mean | 0 / 768 | — | — |

All logged training metrics were finite. The mean next-embedding loss fell by
83.2%, proving that the self-supervised representation objective optimized.
However, final task success was 0%. For 0 successes in 768 episodes, the exact
two-sided 95% upper confidence bound is 0.48%. One 16-episode checkpoint at
50,048 steps produced a single success for seed 9351, but it never repeated and
must not be treated as learned competence.

## Diagnosis

The corrected pipeline is operational, but the current recipe learns visual
dynamics without learning reliable long-horizon manipulation and recovery.
This is not a numerical crash, observation leak, teleport artifact, action
mismatch, or small-evaluation ambiguity. The likely limiting factors are:

1. The pilot uses a training ratio of 32, versus 512 in the upstream visual DMC
   configuration, to fit the first audit window. It therefore performs sixteen
   times fewer representation/control updates per collected step.
2. The agent starts from pixels on the full two-object ordered task with a
   stochastic physical intervention. It gets no successful demonstrations or
   easier grasp/place curriculum before learning recovery.
3. Dense reaching reward can improve without completing grasp, placement,
   ordered suffix recovery, and safety simultaneously. The observed shaped
   return fluctuations without sustained success match this failure mode.

The next high-confidence experiment should retain the same non-teleport
environment and blinded observation contract, but use a within-environment
curriculum (single-object grasp/place, ordered two-object placement, then
physical intervention), a substantially higher train ratio, and DreamerV3 as
an architecture-matched reconstruction ablation. Final claims require a new
multi-seed evaluation; the state-PPO result must remain the current competent
baseline until vision success is demonstrated.

For allocations that reach Jarvis's 24-hour limit, `latest.pt` plus
`replay_latest/` resumes the exact newest model, optimizer, scheduler, scaler,
and replay state. A distinct `best.pt` is selected lexicographically by held-out
success and then return for evaluation. Training deliberately resumes from
`latest`, not `best`, because rolling the optimizer and replay back to an older
evaluation checkpoint would not be an exact continuation.
