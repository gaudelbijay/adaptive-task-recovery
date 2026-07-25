---
title: Simulation Environment — ManiSkill3
status: draft
last_updated: 2026-07-24
---

# Simulation Environment — ManiSkill3

## 1. What ManiSkill3 gives you

- SAPIEN-based rigid-body physics with GPU vectorization: thousands of parallel environment instances on a single GPU, which matters enormously for RL sample efficiency on recovery policies (you need lots of failure/recovery episodes).
- A `BaseEnv` / `BaseAgent` API for defining custom robots (via URDF/MJCF) and tasks; existing tasks and agents to learn the API from before writing custom ones.
- Built-in rendering (rasterization + optional ray-tracing) for RGB-D observations, and access to privileged simulator state such as exact object poses and contact forces, which enables clean failure labels (see [06](06-failure-taxonomy-and-detection.md) §3).
- Randomization hooks for physical parameters (mass, friction, restitution) and initial state, which is the substrate for both standard domain randomization and this project's *failure injection*.

Verify exact current API names (`BaseEnv`, `BaseAgent`, registration decorators, action-mode options) against the installed version's docs/source when you start implementation — simulator APIs move between releases; don't hardcode assumptions from these notes into your mental model of "how it definitely works."

## 2. Getting the Unitree humanoid into ManiSkill3

1. **Source the URDF/MJCF**: Unitree publishes robot description files (URDF, sometimes MJCF) for G1/H1 in their own open-source repos (e.g., robot description packages used by `unitree_ros`/`unitree_mujoco`/`unitree_rl_gym`-style projects). Confirm current locations and licenses directly from Unitree's official GitHub org at implementation time rather than trusting a remembered path.
2. **Import as a custom `BaseAgent`**: define joint names, actuator config (position/velocity/torque control mode and model limits), and mount/collision geometry per ManiSkill's agent-definition pattern.
3. **Sanity-check standing stability first**: before any task or failure work, verify the imported robot can stand still under a simple PD/joint-position controller for N seconds across randomized start joint noise. This is your "does the asset import even work" smoke test — expect to spend real time here tuning joint damping/stiffness and collision margins; humanoid asset import is consistently one of the fiddlier parts of this kind of project.
4. **Add a nominal locomotion/manipulation controller** (either import an existing trained policy if the community has published one for this robot in this sim, or train a minimal walking/reaching baseline yourself) before layering failure work on top — you need *something* that fails in interesting ways.

## 3. Task environments to build (v1 suite)

| Task | Description | Primary failure modes exercised |
|---|---|---|
| `PickPlaceRecovery` | Bimanual or single-arm pick of a tabletop object, transport, place at target | grasp slip, occlusion, misplacement from perturbed pose |
| `DoorOpenRecovery` | Grasp handle, pull door open | contact loss (hand slides off handle), unexpected door resistance |
| `CarryWalkRecovery` | Hold an object while walking a short path | whole-body coordination failure, balance disturbance while manipulating |
| `PushRecoveryStand` | Standing balance under randomized external pushes (no manipulation) | pure balance/locomotion failure — isolates the balance-recovery skill from manipulation |

Build `PushRecoveryStand` **first** — it's the simplest environment (no object interaction) and lets you validate the failure-injection + detection + recovery pipeline end-to-end before adding manipulation complexity.

## 4. Failure injection API (design sketch)

```python
class FailureInjector:
    """Applied inside env.step() at a configurable rate/schedule during training and eval."""

    def maybe_inject(self, env, step_idx: int) -> Optional[FailureEvent]:
        ...

# Example failure types to implement, roughly in order of implementation effort:
# 1. external_force: apply a randomized force/torque impulse to torso or held object
# 2. friction_drop: temporarily/permanently lower friction coefficient of a grasped object
# 3. object_perturbation: teleport/nudge a manipulated object's pose mid-task
# 4. sensor_dropout: zero-out or freeze camera/proprioceptive input for N steps
# 5. actuator_fault: clamp torque limit on a randomly chosen joint
# 6. contact_loss: forcibly break a contact constraint (e.g., simulate hand slipping off a handle)
```

Design goals for this API:
- **Deterministic given a seed**, so failure scenarios are exactly reproducible for evaluation/comparison across model versions.
- **Ground-truth labeled**: every injected failure records type, onset step, and (for evaluation) the ideal recovery window, enabling exact detection-latency and precision/recall measurements (see [10](10-evaluation-and-benchmarks.md)).
- **Severity-parameterized**: each failure type takes a severity scalar so you can build curricula (train on mild, evaluate generalization to severe) and report a severity-vs-success-rate curve, not just one number.

## 5. Domain randomization plan

Randomize across training: object mass/friction/size within task-plausible ranges, robot joint damping/stiffness (±10–20%), sensor noise (Gaussian on joint encoders and IMU), visual properties (lighting, textures, camera pose jitter) if using RGB observations, and action/observation latency. These variations test robustness beyond one fixed simulator configuration.

Keep a **fixed, un-randomized "canonical" eval config** separate from the randomized training distribution, so you always have an apples-to-apples number across experiment iterations in addition to the randomized-robustness numbers.

## 6. GPU parallelization considerations

- Batch environment count is your main lever on RL wall-clock time; profile to find the largest batch size your GPU memory allows for the chosen observation modality (state-only observations parallelize far better than RGB-D — consider training detection/recovery primarily on privileged/state observations first, and treating vision-based observation as a later, harder iteration).
- Keep an eye on CPU-side bottlenecks (Python overhead, logging, dataset writes) once GPU step time gets fast — a common failure mode in GPU-parallel RL projects is spending all your wall-clock time outside the simulator.
- Record steps/sec and GPU memory for each environment config in [13-experiment-log-template.md](13-experiment-log-template.md) — this is exactly the kind of systems-performance detail that stands out in a portfolio writeup versus "I ran PPO."
