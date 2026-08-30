# Visual recovery: staged hypotheses and decision gates

The final policy must execute continuous robot commands in `LearnedRecovery-v3`.
It receives RGB, Panda joint position/velocity, robot TCP pose, and the parsed
two-token order instruction. TCP is robot proprioception derived from its own
kinematics, not object/task state. It never receives cube, goal, sweeper,
protected-object, or task-progress state. Simulator state may be used by a
disclosed training-only critic or by evaluation metrics. The exploratory direct
gate additionally excludes TCP and therefore remains a stricter baseline.

## Predeclared hypotheses

- **V1 — visual control competence:** with no intervention, RGB PPO solves the
  ordered two-object task. Gate: pooled success at least 70% over three seeds.
- **V2 — privileged training helps without inference leakage:** an asymmetric
  critic improves pooled success over symmetric RGB PPO. Gate: positive paired
  difference with a 95% bootstrap interval excluding zero.
- **V3 — temporal representation learning helps:** action-conditioned latent
  prediction improves held-out success over otherwise matched RGB PPO. Gate:
  at least 5 percentage points pooled or a positive paired 95% interval.
  Training-stream learning curves are reported as optimization diagnostics,
  not as an alternative route to hypothesis confirmation.
- **V4 — visual adaptive recovery:** after competence transfer, intervention-
  trained vision exceeds no-intervention training on first-goal-removed cases.
  Gate: positive paired 95% interval over at least 768 held-out episodes.
- **V5 — competitive safe recovery:** the final adaptive visual method matches
  or beats the matched V3 adaptive state PPO in raw and safe success without
  exceeding its violation rate. The primary comparison trains both methods on
  the originally locked 50% intervention distribution and evaluates both on
  identical step-0 physical-removal seeds; every reported episode must contain
  recognized unavailability. Thresholds are read from the frozen state cohort,
  never copied from the earlier forced-sweeper condition. A separate post-audit
  state policy trained directly on the strict distribution reaches 98.44% raw
  and safe success with zero violations. It is a stronger distribution-matched
  extension and the paper's required upper baseline, but it cannot rewrite the
  preregistered V5 primary comparison.

V1 is tested first. V2/V3 are a factorial comparison at the same 40M-step
budget, environment distribution, seeds, optimizer, and evaluation schedule.
Failure of V1 blocks claims about recovery and triggers curriculum/teacher
distillation rather than spending full recovery budgets on an incompetent
controller. All final claims require held-out deterministic evaluation and
per-seed reporting; training-time best metrics are only a selection diagnostic.
The executable verdict configuration is
`configs/visual_recovery_hypothesis_validation_v1.json`. V1, V2, V3, V4, and
V5 each name a primary comparison. Later protocol extensions are included in
the generated report but cannot overturn a rejected primary hypothesis. This
prevents choosing whichever completed ablation happens to produce the most
favorable conclusion.
In particular, V1's primary method remains direct RGB PPO. The later DAgger +
privileged-training policy has its own fallback-competence result: it may
justify proceeding to recovery, but it cannot retroactively turn a failed
direct-RGB V1 test into a confirmation.

## Frozen V1--V5 verdicts

All required three-seed, 768-episode primary inputs are now complete. V1 is
**rejected**: direct RGB PPO reaches only 2/768 nominal raw and safe successes
(0.26%). V2 is **rejected**: asymmetric training reaches 0/768 and does not
improve on direct RGB. V3 is **rejected**: adding temporal prediction to the
direct asymmetric policy also reaches 0/768; the favorable DAgger comparison
is retained only as a protocol extension because its paired hierarchical
interval crosses zero. V4 is **rejected**: adaptive V7 is 8.56 points worse
than clean V6 on first-goal-physically-removed safe success, with paired 95%
interval [−36.36, 20.66]. V5 is **confirmed only for its preregistered primary
comparison**: V7 exceeds the historical state reference under identical strict
evaluation seeds. That primary reference is only 2.86% safe and is superseded
for paper competitiveness by the post-audit strict-trained state PPO at 98.44%
safe with zero violations. Thus V5 does not establish state-of-the-art or
distribution-matched competitiveness. The authoritative generated artifact is
`results/final_visual_comparison/hypotheses.json`; later extensions cannot
rewrite these verdicts.

## Post-registration integrated-policy hypotheses

The rejected V1--V5 primaries remain immutable. Later experiments answer three
separately labeled questions under the corrected strict-removal and nominal
protocols:

- **I1 — integrated restricted-input control:** one RGB-deployment policy must
  achieve at least 90% strict and nominal safe success, at least 85% safe
  success on each physical-removal branch, and at most 5% violations in both
  regimes. **Confirmed in the three-seed screen by V19.** It reaches 96.35%
  strict safe, 91.41% nominal safe, 97.06%/95.69% branch safe, and
  1.30%/3.65% violations. The worst endpoint is 91.41%. This is a post-hoc
  dual-specialist student with privileged training, not a revision of V5.
- **I2 — full-strength anti-collapse SSL improves integrated control:** V20
  must outperform otherwise matched V19 under the same selector while retaining
  eligibility. **Rejected.** V20 improves matched-pixel pose R² by +0.0106
  [0.0016, 0.0212] and goal-resolution R² by +0.0146 [0.0010, 0.0377], but
  falls to 85.42% strict safe and 74.06% first-removal safe success. This is
  evidence that improved linear decodability need not improve control.
- **I3 — V19 is robust to optimizer randomness:** retain all five fixed seeds
  and repeat both regimes and removal branches without relaxing I1 thresholds.
  **In progress.** The three-seed gate passed and released fixed new seeds
  71064 and 84293; no confirmation seed may be filtered.

V21 asks a narrower scale hypothesis fixed before V20's held-out result: does
reducing only the VICReg variance coefficient from 0.01 to 0.001 restore I1
eligibility and beat V19's worst endpoint? **Rejected.** Its exact three-seed
chain reaches 92.19% nominal safe success but only 87.63% strict and 78.34%
first-removal safe success; the frozen selector therefore retains V19. V24 and V25 are
separately disclosed post-hoc action-consistency tests and cannot change the I2
verdict. Both are now rejected before full allocation: V24 failed four stability
checks, while scaled V25 passed five of six but missed its frozen best-score
margin by 0.47 points. Neither has held-out or robustness evidence.

- **I4 — V19 continuation-stage temporal SSL contributes to robust control:**
  compare V19 against V26, an exact control changing only the continuation
  temporal coefficient from 0.01 to 0.0. Confirm only if V19 improves the
  control's worst strict/nominal/removal safe endpoint by at least 3 points and
  the paired hierarchical 95% interval has a positive lower bound at that
  limiting endpoint, while V19 remains below 5% violations in both regimes.
  **Not confirmed.** V26 reaches 90.49% nominal and 93.88% strict safe success.
  Its limiting endpoint is nominal, where V19's gain is only 0.91 points and
  the paired hierarchical 95% interval is [-4.04, 6.51], failing both the
  three-point effect and positive-lower-bound requirements. Both arms inherit
  upstream V6/V13 checkpoints trained with temporal SSL and privileged
  supervision, so I4 isolates the continuation loss only and cannot establish
  a fully SSL-free lineage comparison.

- **I5 — selected visual control is robust to frozen sensor/render shifts:**
  require at least 75% safe success and at most a 15-point paired safe-success
  drop for every pixel, color, camera, and lighting variant in both nominal and
  intervention conditions. **Rejected for the independently selected V19
  incumbent.** Every OOD variant fails at least one joint criterion; 4-pixel
  translation and +5 cm camera height fall to 5.08% and 2.86% intervention safe
  success. A post-hoc V27 generic shift/color self-distillation smoke retained
  its 85% in-distribution floors but improved matched-seed mean OOD by only 4.69
  points, left worst OOD at 0%, and was rejected before three-seed allocation.
  Neither result changes I1; they bound it to the declared visual domain and
  motivate actual rendered-domain training plus a new unseen confirmation suite.
  V28 was that first post-hoc test. It improved mean matched OOD by 31.14
  points but failed nominal retention (82.81% vs 85%) and worst OOD (0% vs
  25%), so it was rejected before three-seed allocation. V29 is a disclosed
  observed-suite tuning response that freezes V19's policy/progress heads and
  trained only its encoder with teacher-feature anchors plus rendered and exact
  sensor shifts; it repaired nominal retention but failed intervention and
  worst-OOD gates. V30 tested full-episode multidomain DAgger, but its
  mismatched state teacher overwrote V19 and failed six gate checks. V31
  rendered three simultaneous cameras from one physical state with V19 as its
  sole teacher. It improved matched mean OOD by 26.20 points and retained
  nominal behavior, but failed intervention retention and worst-OOD checks.
  V32 then preserved V19 on a learned RGB in-domain route and trained a
  geometry-grounded adapter; it retained nominal/intervention behavior but
  failed mean and worst-OOD thresholds. V33 learned paired RGB canonical-view
  synthesis. It passed retention, causal, and mean-improvement checks but was
  rejected at 0% worst OOD and a -26.17-point worst regression. Its forced-route
  diagnostic confirms that pixel failure reflects both routing and synthesis,
  motivating V34's factorized learned dense warp plus photometric residual.
  V34 remains observed-suite tuning until it passes the frozen smoke gate; the
  D-168 confirmatory suite remains unavailable during design and training.
  separately geometry-grounded RGB adapter for detected shift. I5 remains
  rejected unless a
  three-seed candidate passes
  standard/strict retention and the separately frozen right/low-camera,
  bright/cool-lighting, opposite pixel/color suite. Twenty-step paired segments
  and privileged labels prohibit full-episode, pure-SSL, or RL claims.

The three predeclared training seeds are a screening experiment, not unlimited
evidence about optimizer randomness. Reports include pooled Wilson intervals
(conditional held-out episode uncertainty) and a hierarchical bootstrap that
resamples trained policies and then episodes within each policy. Paired effects
use matched training seeds and matched held-out episode seeds with the same
hierarchy, following the few-run evaluation concerns in
[Agarwal et al.](https://arxiv.org/abs/2108.13264). The original V7/V8
forced-sweeper confirmation chain was retired before allocation after the
strict-removal audit showed that endpoint rarely produced actual removal. The
post-audit V13 extension receives two new from-scratch seeds only if the same
checkpoint passes all six frozen integrated checks: strict and nominal safe
success, both strict removed-goal branches, and strict and nominal violation
limits. This extension confirmation cannot rewrite a preregistered V5 verdict;
screening seeds are never selectively discarded.
Those seeds are fixed as **71064** and **84293**, the first two unique integers
produced by `numpy.random.default_rng(20260828).integers(1, 100000)` after
excluding the three screening seeds. They were generated before the three-seed
recovery aggregate existed. Confirmation retrains the full clean DAgger+PPO
initializer and V13 integrated visual policy, plus the matched integrated-
mixture state PPO, from scratch. A strict combiner requires all five visual and
state seeds and 256
held-out episodes per seed; neither weak confirmation seed can be dropped.
Visual and state evaluation use the same seed base, training-seed identifiers,
episode count, and reset schedule. Because the linear reset formula for these
large training-seed identifiers exceeds ManiSkill's legacy RNG range, both
evaluators use the same documented SHA-256-to-31-bit fallback only for
overflowing batches; screening reset seeds remain unchanged, and exact batch
seeds are stored and recomputed by the aggregate. The final comparison verifies
intervention and instruction branches episode by episode before computing
hierarchical paired visual-minus-state effects; a mismatch is a hard error, not
silently treated as paired data.
Sample-efficiency tables report PPO environment steps, online BC/DAgger
transitions, and their sum separately. DAgger interaction is never hidden by a
checkpoint counter that begins at PPO step zero.

The learned-progress DAgger fallback has now passed its competence screen:
97.40% nominal raw success and 96.48% nominal safe success over 768 held-out
episodes. It also achieves 92.19% raw and 91.02% safe success under the forced-
sweeper condition, with 1.43% violations. Only 125/768 episodes produced
recognized physical unavailability, including just five actual first-goal
removals, so those pooled values are not labeled post-removal recovery. A
separate step-0 strict-removal protocol requires actual unavailability in every
episode and is never pooled with the original condition. The clean visual
cohort passed that invariant in 768/768 final episodes and achieved 52.60% raw
and 52.34% safe success with 0.65% violations. V4 now reads only the paired
first-goal-physically-removed contrast from this strict aggregate; V5 reads
only the matched strict visual-versus-state comparison. Target-only branch
labels cannot enter either verdict. The authorized adaptive V7 continuation is
now complete. It retains 94.14% nominal safe success but achieves only 32.42%
strict safe success (249/768), below the clean cohort's 52.34%; its first-goal-
removed safe rate is 20.59% versus 29.14% clean, and the paired interval crosses
zero. This does not confirm V4, while V5 still requires the frozen final method
and confirmation protocol. The clean policy's matched identical-pixel pose
probe is negative: learned-minus-random variance-weighted R² is −0.177 with
seed-bootstrap interval [−0.334, −0.037]. V7's separate probe is positive at
+0.387 [0.312, 0.488], but V7 combines temporal SSL, privileged pose auxiliary
supervision, and supervised progress labels. Accordingly, the V7 result is
reported only as linear decodability; it is not evidence that temporal SSL
caused the gain or improved control.

Qualitative README media is also gated rather than treated as free-form
cherry-picking. The capture script searches the first safe success in a fixed
declared seed range, but it may run only after the same method's three-seed,
768-episode aggregate matches the V3 state reference in raw recovery, safe
recovery, and violation rate and retains at least 70% nominal success. Its JSON
records the search range and selected seed. A visually appealing isolated
success from a weaker aggregate is ineligible to replace the README montage.

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
Because teacher-forced behavior cloning can suffer autonomous covariate shift,
a matched DAgger variant linearly increases student-executed actions to 0.8
while retaining teacher labels. Plain BC and DAgger remain separate candidates;
the initial autonomous safety failure of plain BC is retained as negative evidence.
The learned-resolution DAgger branch uses one frozen expert checkpoint for all
student seeds. Teacher seed 1788 was selected before this branch started because
its already-completed clean nominal evaluation was 256/256, versus 125/256 and
17/256 for the other available teachers. This is disclosed expert-quality
selection using state-policy validation data; final visual checkpoints and
held-out visual episodes were not used for it.
This was a sequencing-expert choice, not a claim that the teacher is a safety
oracle: seed 1788's nominal safe success was 52.0% and its full-horizon
violation rate was 48.0% (versus 42.2%/7.0% for seed 4796). Consequently the
student must reduce violations through the safety-penalized PPO phase and still
pass V5's matched V3 state-reference violation gate. Raw imitation success
alone cannot confirm the method; the historical 8.59% V2 rate is context, not
the V3 acceptance threshold.

A separate privileged-representation candidate predicts scaled cube/sweeper
positions and task progress from the RGB latent with loss weight 0.1. These
simulator labels train only an auxiliary encoder head and are absent from actor
inputs and evaluation. This candidate is explicitly labeled privileged visual
training, not self-supervision; temporal prediction remains the SSL component.

Linear-probe comparisons use a frozen seed-matched behavior checkpoint to
collect the probe trajectories. Learned encoders, matched random encoders, and
all compared methods therefore receive byte-identical RGB and label tensors;
the files record SHA-256 digests and the final comparator fails on any mismatch.
This removes policy-induced observation distribution as a confound. Pose labels
are used only by the post-training linear analysis, so probe R2 is evidence of
decodability, not proof that the controller causally uses that representation.

The strict feed-forward student reached roughly 0.53--1.00 completed goals per
episode but did not transition reliably to the second goal. This is a post-hoc
diagnosis, not a predeclared confirmation. The follow-up candidate predicts two
ordered goal-resolution bits from the RGB latent and feeds those predictions to
the actor. A bit is positive after valid placement or causally recognized
physical removal. Ground-truth resolution supervises that predictor during
training only; held-out execution computes it exclusively from pixels. We
report its complete confusion matrix, positive/negative recall, balanced
accuracy, target and predicted-positive prevalence, bit accuracy, and exact-
vector accuracy. This exposes majority-class collapse that raw accuracy can
hide. The method is labeled privileged visual training, not pure self-
supervision. It can support a restricted-input visual deployment claim, but
not a claim of training without simulator supervision.

Only a full ordered-task checkpoint is eligible to initialize adaptive recovery.
Recovery training samples physical interventions with probability 0.5 and model
selection evaluates probability 1.0. Final held-out reports include both forced
intervention and nominal conditions so adaptation and retention are separable.

Before any V3 clean held-out aggregate existed, the adaptive follow-up was
extended to the matched no-progress DAgger pair. One candidate retains the
pose auxiliary but no temporal loss; the other differs only by temporal SSL
coefficient 0.05. Each three-seed 100M-step recovery array is held until its
own clean method reaches at least 70% pooled success over the predeclared 768
held-out nominal episodes. This is a disclosed protocol extension motivated by
the progress head's weak training-stream semantic accuracy, not part of the
original V1--V5 preregistration. It permits a direct adaptive no-SSL versus
temporal-SSL comparison and prevents the final self-supervision conclusion from
depending solely on a policy trained with privileged progress labels.

## Post-hoc controller-retention gate

The pixel-resolution branch was introduced after observing the strict policy's
second-goal bottleneck. A further intervention is allowed only if its pooled
training-time full success remains below 5% at 4.9M PPO steps. The predeclared
follow-up is a **kickstarting ablation**: retain the same RGB actor, fixed
teacher, DAgger data budget, environment, and held-out protocol; reduce the
post-BC resolution auxiliary coefficient from 1.0 to 0.1; and add a teacher
action MSE term that decays linearly from 0.1 to zero over the first 5M PPO
steps. Ground-truth state and teacher actions remain training-only. The
follow-up is useful only if it improves pooled 4.9M success by at least five
percentage points without increasing violations; otherwise it is rejected.
This gate is explicitly post-hoc and cannot be presented as a predeclared V1--V5
confirmation.

The later reward audit superseded this V2-only gate before a kickstarting job
was launched. Because all V2 candidates optimize the same success-delaying
reward, their 4.9M comparison cannot distinguish controller forgetting from
objective misalignment. The current V2 runs and their gate measurements remain
negative diagnostics, but no additional GPU budget is assigned to the V2
kickstarting or intervention branches. V3 begins from the already-declared
DAgger candidate without a result-contingent kickstarting change.

At the analogous 4.907M V3 checkpoint, the three temporal+learned-progress
policies achieved 89.06%, 87.50%, and 90.62% full success with zero
training-evaluation violations. No V3 kickstarting fallback is therefore
triggered. Progress-bit accuracy at that checkpoint was only 61.85%, 47.38%,
and 55.99%, so the snapshot supports restricted-input visual control but does
not confirm that the auxiliary head learned a semantically faithful task-state
representation. That mechanism requires the held-out progress audit and the
temporal/no-progress factorial arm.

The initially observed weakness of the temporal/no-progress arm triggered a
code-path audit, not a result-contingent change to that running cohort. The
temporal predictor receives the exact bounded action executed by the policy and
the current RGB latent; its stop-gradient target is encoded from the immediate
post-action RGB observation. Transitions ending an episode are masked with
`1 - next_done`, so auto-reset images cannot become temporal targets. A source
contract test fixes those invariants. No target-alignment or reset-leakage defect
was found, so a weaker final temporal result will be reported as negative
evidence rather than silently replaced or relabeled.

The later V16 anti-collapse extension is explicitly post-registration. It
retains the V15 RGB actor, state-teacher data, seeds, PPO budget, and train/eval
distributions while adding VICReg variance and covariance penalties to the
online visual latent. A constant latent incurs the variance penalty and
cross-coordinate correlation incurs the covariance penalty. V16 must pass an
operational smoke before full allocation and is evaluated against V15 on both
held-out control and byte-identical-pixel linear-probe endpoints. A probe gain
does not establish a causal control gain, and a control gain does not by itself
establish a better general-purpose representation.

The pose probe is complemented, not replaced, by a separately versioned
task-semantic probe. A frozen behavior policy generates byte-identical pixels
and ordered goal-resolution labels for every compared encoder. Linear ridge
heads report macro balanced accuracy, macro ROC AUC, and R² for the two bits;
matched random encoders expose information available from an untrained visual
projection. Since the policy's progress head is itself trained with
simulator-derived resolution labels, this diagnostic measures downstream task
state decodability and can isolate V16-minus-V15 effects, but it is not evidence
of label-free representation learning by the complete system.

The completed held-out factorial favors temporal SSL in pooled performance but
retains substantial seed uncertainty. Without temporal SSL, nominal success is
611/768 (79.56%) and forced-sweeper-condition success is 620/768 (80.73%).
With temporal SSL, the matched values are 747/768 (97.27%) and 704/768
(91.67%). Temporal-minus-control differences are +17.71 nominal and +10.94
forced-sweeper percentage points. Their hierarchical paired intervals include
zero because the no-SSL arm is highly seed-sensitive, so this is supportive
pooled evidence, not a claim of a uniformly positive seed-level effect. It is
also the protocol-extension comparison for V3 and cannot overturn the still-
pending primary direct-RGB factorial verdict.

`LearnedRecovery-v2` closes an exploit in the original benchmark: an object is
skippable only when it is the preselected physical intervention target, is
actually unavailable, and intervention onset has occurred. Objects thrown away
by the robot do not satisfy a goal. Runs started under v1 semantics are retained
as exploratory diagnostics and are ineligible for final claims.

### Reward-objective audit and V3 correction

Before held-out visual evaluation, a code-level reward audit found that V2 paid
`3 * completed_goal_count` on every remaining step. With the configured
normalization and discount (`gamma=0.95`), stalling after the first goal can
produce `0.3 / (1 - 0.95) = 6.0` discounted reward, whereas completing the
second goal produces a terminal normalized reward of 1.0. The consistent
approximately-one-goal training trajectories are therefore not evidence of a
representation-only limitation: the objective itself rewards delaying success.

`LearnedRecovery-v3` is an isolated subclass and leaves all V2 scene geometry,
physics, observations, continuous control, intervention, ordering, safety, and
success semantics unchanged. It replaces persistent state occupancy rewards
with bounded step-to-step reaching/grasping/placing progress, a one-time goal
completion bonus, and the existing terminal success and safety terms. An
unchanged state away from the protected object receives zero task reward, so
waiting after goal one cannot accumulate return. V2 jobs are retained as a
fully disclosed negative reward-audit cohort, but no V2 result is eligible for
V1--V5 confirmation. All final state and visual methods are rerun on V3 with
matched seeds and budgets. This correction was declared before any V3 training
or held-out result existed.

The subsequently completed three-seed V2 state-policy audit makes the failure
mode explicit. On 768 forced-intervention episodes, where one object is often
physically removed, raw success was 50.65% and mean completed goals was 0.924.
On the matched 768 nominal episodes requiring both placements, success was only
1.95%. Thus the apparently strong V2 intervention score primarily measures the
one-feasible-goal branch and is not evidence of reliable ordered two-goal
control. These V2 held-out results were produced after V3 was specified and do
not influence its reward coefficients or training configuration.

The matched V3 state-policy reference subsequently completed its predeclared
held-out evaluation. Across 768 forced-intervention episodes it achieved
425/768 raw successes (55.34%, Wilson [51.81%, 58.82%]), 424/768 safe
successes (55.21%, Wilson [51.67%, 58.69%]), a 1.56% constraint-violation
rate, and 1.010 mean completed goals. Per-seed raw success was 63.67%, 50.78%,
and 51.17%; the corresponding violation rates were 0%, 3.91%, and 0.78%.
Its matched nominal evaluation was 0/768 with a 0.13% violation rate. The
forced-intervention result is therefore the declared V5 recovery threshold,
not evidence of nominal two-goal competence. The separately trained nominal
state cohort supplies that control-solvability upper baseline.
