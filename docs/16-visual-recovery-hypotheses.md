# Visual recovery: staged hypotheses and decision gates

The final policy must execute continuous robot commands in `LearnedRecovery-v1`.
It receives RGB, Panda joint position/velocity, and the parsed two-token order
instruction. It never receives cube, goal, sweeper, protected-object, TCP, or
task-progress state. Simulator state may be used by a disclosed training-only
critic or by evaluation metrics.

## Predeclared hypotheses

- **V1 — visual control competence:** with no intervention, RGB PPO solves the
  ordered two-object task. Gate: pooled success at least 70% over three seeds.
- **V2 — privileged training helps without inference leakage:** an asymmetric
  critic improves pooled success over symmetric RGB PPO. Gate: positive paired
  difference with a 95% bootstrap interval excluding zero.
- **V3 — temporal representation learning helps:** action-conditioned latent
  prediction improves held-out success or sample efficiency over otherwise
  matched RGB PPO. Gate: at least 5 percentage points pooled or a positive
  paired 95% interval.
- **V4 — visual adaptive recovery:** after competence transfer, intervention-
  trained vision exceeds no-intervention training on first-goal-removed cases.
  Gate: positive paired 95% interval over at least 768 held-out episodes.
- **V5 — competitive safe recovery:** the final visual method matches or beats
  the state PPO reference (51.69% safe success, 59.77% raw success, 8.59%
  violations) under the same intervention protocol and episode budget.

V1 is tested first. V2/V3 are a factorial comparison at the same 40M-step
budget, environment distribution, seeds, optimizer, and evaluation schedule.
Failure of V1 blocks claims about recovery and triggers curriculum/teacher
distillation rather than spending full recovery budgets on an incompetent
controller. All final claims require held-out deterministic evaluation and
per-seed reporting; training-time best metrics are only a selection diagnostic.

The predeclared curriculum fallback changes only the number of goals required
for termination during pretraining. It uses the same scene, camera, robot,
continuous controller, reward components, randomization, and physical dynamics.
The transferred policy is then optimized and evaluated with both ordered goals
required; no curriculum result is reported as full-task success.

A separate teacher-distillation fallback reconstructs the existing state PPO
input from named training-only fields and behavior-clones its bounded commands.
The RGB student is subsequently fine-tuned with PPO. This is reported as
privileged training, never as pure visual RL; deployment and held-out evaluation
still instantiate only the restricted visual actor.
