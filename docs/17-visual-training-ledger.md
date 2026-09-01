# Jarvis visual training ledger

> **Naming.** This is a per-candidate ledger, so the counter is the primary key
> and is retained deliberately. Row titles of the form "V19 <name> V27" read as
> "frozen base policy / candidate built on it". See
> [`31-naming-and-identifier-key.md`](31-naming-and-identifier-key.md).


This ledger separates exploratory jobs from experiments eligible for final
claims. Times are Jarvis scheduler times in America/New_York. All arrays use one
L40S per task, atomic checkpoints, and the repository's 24-hour continuation
wrapper.

## Latest V41 chain

The exact three-seed V36→V38→V40 (the audited canonicalization lineage) lineage completed and passed each immutable
audit as jobs `1144450`--`1144455`. Standard and strict evaluation jobs
`1144458`--`1144461` then completed across all three seeds. The 27-task
untouched array `1144462` ran at up to eight concurrent L40S allocations;
aggregate `1144463` completed, and fail-closed gate `1144465` wrote its result
before exiting 1 on unmet thresholds. No task failed operationally.

The immutable outcome is 89.45% standard nominal, 95.57% standard
intervention, 96.35% strict, and 44.47% mean untouched safe success. The gate
passes 6/10 checks. V19 (the dual-specialist RGB controller) remains the released controller; V41 (the magnitude-gated canonicalization controller) is retained as a
positive geometric-canonicalization mechanism result with unresolved camera
and directional-light robustness.

The later V44--V52 (the hierarchical-routing candidate) structural sequence culminated in V52 passing all six
opened development checks, then failing its separately reserved seed-127M
confirmation with a 0% minimum cell. Seed-127M became opened development data;
seed-133M was frozen before any successor rendering. V53's (the renderer-family repair candidate) renderer experts
completed as jobs `1144698`--`1144700` and retained 90.23%/93.75%
nominal/intervention, but mean/worst opened OOD were 50.59%/0%.

V54 (the continuous geometry-corrector candidate) jobs `1144703`--`1144705` completed without an operational failure. The
trainer consumed exactly 800,000 synchronized transitions, but the frozen
five-way route produced rollout results byte-for-metric identical to V53: its
74.0% final router accuracy never produced a behaviorally effective route at
confidence 0.90. V54 is rejected while its trained correctors are retained.
The independently frozen V55 binary router completed 480,000 transitions in
job `1144783` with 80.27% final-100 accuracy. Tensor-only V56/V57 composition
jobs `1144794`/`1144798` passed; development arrays `1144795`/`1144799`,
aggregates `1144796`/`1144800`, and gates `1144797`/`1144801` are live. These
are opened-suite routing controls, not multi-seed or untouched evidence.

Both controls were rejected: V56 passed 3/6 and V57 4/6. V58 then forced V54
specialist selection behind V39's detector and regressed nominal control to
69.53% (2/6). V59 restored exact V39 geometry control under V53 renderer
routing, reaching 92.97%/96.09% nominal/intervention but missing mean/worst OOD
at 63.87%/5.86%. V60's (the residual-corrector composition controller) targeted residual composition completed as jobs
`1144838`--`1144841` and passed all six checks: 68.48% mean OOD, 30.08% worst
OOD, 92.97%/96.09% retention, and a 30.08-point causal drop with positive lower
bound. Combined-similarity nominal safe success rose to 42.58%.

V60 is rejected and not released. The complete independent lineage for seeds
`[9351, 4796, 1788]` completed as jobs `1144860`--`1144869`; the manifest
hash-pinned all ten base stages and rewrote only seeds, output names, and
declared source paths. Standard/strict arrays `1144903`/`1144904` then produced
83.46%/93.36% pooled safe success. Seed 9351 collapsed to 69.53% nominal safe
success with 12.50% violations. The reserved seed-133M suite was not opened.
This closes V36--V60 as a controller-patching line; D-230 records the rejection.

V42 (the unconstrained renderer-repair candidate) train/audit/development/gate jobs `1144499`--`1144503` completed and were
rejected at 2/6 checks after clean-route collapse. The sole bounded V43 (the identity-protected renderer-repair candidate) repair
`1144513`--`1144517` completed and passed 4/6: retention and causal checks pass,
but mean/worst development OOD fail. No multi-seed or untouched task was
allocated for either version. These results close further coefficient tuning
of the same dense pixel-reconstruction mechanism.

The local and Jarvis launch snapshots match byte-for-byte. Core SHA-256
provenance for the V3 cohorts is:

- base environment: `221917d8bd5088a7f5b79a5389eee2f41dbfac3b86db8abf3423dd4934a54839`
- V3 event-reward environment: `272c230d4b2638d4cb9d9bc913ea947c66ac41a02e005956d79ec5e7c54ee422`
- visual trainer: `2cb6a21f694a6881b4c37e7060caaa16c24613819a8847eb951483bfe08de136`
- VICReg visual trainer: `266f60b4059c16e8daf180339c533d1fe5c27d12d521a333f3b9d4b29ba89689`
- state trainer: `9ac83f8af68e537fc6fd0de14731994659127956a99a40bb9117d0d43365d3fd`
- state continuation trainer: `871ba7464ae95980152483e2a08937a98075d020aa75889f1305df47a69ed4b2`
- visual held-out evaluator: `a9cc42382ea4e6b3c3a99dee52ad2e5f00f5912009bc7e64e42677b7f5e67671`
- state held-out evaluator: `28f3755b8cf31e2c8ef40c8c8b8bba2e96ce58351143e177f2dd60221ad608a9`
- overflow-safe evaluation-seed derivation: `757fc163339a26b1e5d99d501c738527393035f32fa8cf61319b17bd68e62c97`
- visual aggregate: `137713d4a98b52a47e37d9712bea621d26f45b80cc5fd07c5bbcc2c7cdf143df`
- state aggregate: `c60c2b4556e7e420f6f60cef6aa5bfe488821593ff8dd8bdaa165cf127b5e879`
- final visual/state comparator: `f599d08bf0cefab0c81f9626ac0b60f6360b1da4866954ab336b30a6671d229e`
- final held-out figure renderer: `bd37b6d99cc05e74211a7d14cfafc94fcf3c7eb686979dbc1cd06e6d7c321665`
- training-stream learning-curve renderer: `a23f2d463da6abd3d1c83256ace64ff0b54af72dd7dfdf6ffefcf29a86990325`
- representation probe: `3f0c333bda8b36db68146db28cc7d88c3285d1d276087ca039f606a805890857`
- representation-probe aggregate: `59732ce59ebcc296836091c58e330d033a6c99ac915cc35bdce1b0f25fd59f0b`
- matched-pixel representation comparator: `8fd750b39b6da3f85cd7c3ce96180c266d83a64a9f3f11d7d5bcfefac192fc7d`
- task-semantic representation probe: `4e2eb042f0b948389720dbc6bd76f643517105620143fb428219ebdc39cfe99f`
- task-semantic probe aggregate: `26741e897dff57d73838d9b272109051dc4ae862f1d2541bdf5b2e42a755a3b9`
- task-semantic matched-pixel comparator: `91163a5a1af02506a8be50a2a497201f96995a4d43a535bb74296a9faa6f656f`
- confirmatory V5 gate: `2c63eff56d553f53068fdeb5f2848d77e84cfbe7372a91721b504dfca53f5520`
- five-seed visual/state combiner: `3ba6ad0f7ada11bf0bb57d04f712f0c23d9305979ae2349e6b7d080cd2a5b275`
- five-seed figure renderer: `62f6da95850132f5c38bb8cefb90f1d7908edb52290440f583467494244cd99e`
- competence/media gate: `618ae5253040a9985a833e1fdf2cfa3436cd448fc3889ded5293e6e99dd1e035`
- recovery-video capture: `7d3d513cbc7d4a67908eb6c7c7574bed955d95266f347cc5dc0e0c6ba676995a`
- integrated state-teacher allocation gate: `878ee91e2c4ada646a2ed746fac733e7c105e43bf982a96dbeb84f6310e8c52a`
- method-information contract builder: `3f7cbceaa14ce4ede44120a6445b6a8b10f5c3fd0c7119ed7f43fbec09074958`
- method-information contract config: `9e3daebe75dd33814c8aa0f31eacc35f27de4ebc09196463051e5be758e407e6`
- dual-specialist visual trainer: `e9feca5ddceaa4203e873d1dde2af942d337fc953610b6338672cbfcb2c0a166`
- dual-specialist V19 config: `bd9e5734a09895304a3c6c65d9cd2ba727f526179373eb2056fdf66b7f05face`
- dual-specialist V19 selector: `86bea2ec01ff2e0980468c003dc4645f551efabc5283e5366ba30178c26f1795`
- dual-specialist teacher gate: `63301b6541acdb14eac1c9a7d76e320c285136cb71d6ce7404bf034d9125af33`
- dual-specialist teacher-gate config: `81b26649802571ccfb04bad77a4f1c623c18943b3b33840ac66bd417235ea20c`
- dual-specialist VICReg V20 (the full-strength VICReg ablation) trainer: `c53b95845bb597a0bfecc889bf90620eef53ba3cbfb4fc2bfcbdaf6a6d5f875a`
- dual-specialist VICReg V20 config: `33604faeb7769d8b69012804d038e421e5f5d7493c70719825e95c7098e8261c`
- low-variance VICReg V21 (the low-variance VICReg extension) config: `4e623c9cda90c4fd494e96516d1d2dfe7214cf44ed79d9b39d902b5f2e1030bb`
- dual-specialist V19/V20 selector: `8f3506d50801ab76125b819028d4a0735699c4c5b877c69d07ceb51e79d4d152`
- dual-specialist release router: `eab6e4d53a3a8e8b21a8c9b19f5202fd6ce4ba2c565a8424ca9af3f8d48b6bf3`
- dual-specialist release-router config: `f5f57949e8cf9ffcbb27d84dd186fc183af5bd902a787a4238972dc615078be1`
- state-fallback release router: `5d3de5b157fedc2e58d35572ebca7e2f93c66c1c59ec92c9d543f6d83ee33de5`
- state-fallback release-router config: `7fbe1cb0cd620fa3c160e109f84cffc1f55c60477d45840208594d0c79883e38`
- DrAC-style visual trainer: `357f49c59537d26ceaceec0d45d9538993338209878bb775e3893445e037c755`
- DrAC policy-consistency helper: `00dc44145d48c574f6f5f937d69a66b3349f5e18fffee6a7a6758734e8b74c0b`
- calibrated failure-only DrAC runtime config: `bdb5d9f764ea23a0110d123883edfc3a91e8374cfc89cca75f7f07419ee1f5c7`
- bounded shift-action consistency helper: `3ae0b45cc15b375a6cbfce3595227f7ad41120c283e3a056461c90c5fce8b162`
- bounded shift-action trainer wrapper (full-allocation provenance repair): `77cd6312c6ab9f0515647618963f8b5d2a7f160ea2cbb10fcf419551d6e9434b`
- archived bounded shift-action smoke wrapper: `037c5403aac7a911f32e5bc4b185aa2ebe65604ae2d407ad349e22b6ca23cd39`
- bounded shift-action V24 runtime config: `27141ed54f9840713e3b7800f8f99154c35559d11ce236e9d2f75f4f76aaf50e`
- V1--V5 verdict generator: `0b277c057a040714026f2eab648b1343db955d889583756398122c894eb496fe`

| Stage | Config | Train job | Eval / aggregate | Status | Claim use |
|---|---|---:|---:|---|---|
| Direct visual gate, 3 methods × 3 seeds | `visual_recovery_ppo_gate_v1.json` | `1139228` (tasks 3–8), `1139237` (tasks 0–2) | `1139242` / `1139243` | Stopped at 6.5–7.4M; evaluator cancelled | Exploratory and ineligible; preserved checkpoints, near-zero unstable success |
| First-goal visual curriculum, 3 seeds | `visual_recovery_curriculum_v1.json` | `1139246` | — | Stopped at 3.3–4.1M | Negative result: unstable near-zero success; checkpoints preserved |
| Direct clean visual control, 3 seeds | `visual_recovery_direct_v2.json` | `1139267` | `1139268` / `1139269` | Cancelled before allocation | Pruned after strict distilled policy isolated the missing-progress bottleneck |
| Privileged pose-auxiliary visual control, 3 seeds | `visual_recovery_privileged_aux_v4.json` | `1139339` | `1139340` / `1139341` | Stopped at 9.7--9.9M; evaluator cancelled after reward audit | V2 diagnostic only; checkpoints and logs preserved |
| Ordered two-goal transfer, 3 seeds | `visual_recovery_transfer_v1.json` | `1139247` | `1139255` / `1139256` | Cancelled with failed curriculum gate | No result claim |
| State-teacher visual bootstrap, 3 seeds | `visual_recovery_distilled_v1.json` | `1139252` | `1139253` / `1139254` | Stopped at 16.1--20.0M; evaluator cancelled after reward audit | V2 diagnostic only; checkpoints and logs preserved |
| State-teacher DAgger bootstrap, 3 seeds | `visual_recovery_distilled_dagger_v2.json` | `1139320` | `1139321` / `1139322` | Cancelled before allocation | Redundant after PPO corrected plain-BC safety but strict actor still failed sequential transition |
| Clean-state-teacher DAgger, 3 seeds | `visual_recovery_clean_teacher_dagger_v3.json` | `1139331` | `1139332` / `1139333` | Cancelled before allocation | Pruned because it retained the same missing-progress actor contract |
| Physical-intervention visual recovery, 3 seeds | `visual_recovery_intervention_v1.json` | `1139257` | `1139258` / `1139259` | Cancelled with failed curriculum gate | No result claim |
| Pose-auxiliary physical-intervention recovery, 3 seeds | `visual_recovery_privileged_aux_intervention_v4.json` | `1139342` | `1139343` / `1139344` | Cancelled before allocation after reward audit | V2 objective is ineligible for final claims; configs retained for audit |
| Pixel-predicted-resolution DAgger visual control, 3 seeds | `visual_recovery_progress_dagger_v5.json` | `1139347` | `1139348` / `1139349` | Two seeds stopped at 7.5--7.8M; third stopped during DAgger; evaluator cancelled after reward audit | V2 diagnostic only; checkpoints and logs preserved |
| Pixel-predicted-resolution end-to-end smoke, 1 seed | `visual_recovery_learned_progress_smoke.json` | `1139370` | — | Complete; released `1139347` | BC/DAgger, PPO, SSL/auxiliary losses, checkpointing, and completion sentinel all passed with finite metrics |
| Pixel-predicted-resolution adaptive recovery, 3 seeds | `visual_recovery_progress_intervention_v5.json` | `1139350` | `1139351` / `1139352` | Cancelled before allocation after reward audit | V2 objective is ineligible for final claims; configs retained for audit |
| Pixel-predicted-resolution representation probes, 3 seeds | `visual_recovery_progress_intervention_v5.json` | — | `1139353` | Cancelled before allocation after reward audit | Deferred to the final V3 candidate |
| Final cross-method comparison | `visual_recovery_comparison_v1.json` | — | `1139369` | Cancelled before allocation after reward audit | V2 and V3 results are prohibited from being mixed; a V3-only comparison will replace it |
| Clean-semantics state PPO re-evaluation, 3 methods × 3 seeds | `learned_recovery_ppo_v6.json` | existing checkpoints | `1139262` / `1139263` | Complete | Clean state baselines; original result files remain untouched |
| Fresh clean-semantics state PPO, 3 seeds | `learned_recovery_ppo_v7_clean.json` | `1139327` | `1139328` / `1139329` | Complete | V2 reward-audit reference: 50.65% forced-intervention versus 1.95% nominal success |
| V3 event-reward simulator smoke | `visual_recovery_event_reward_smoke.json` | `1139377` | n/a | Complete | Full BC, PPO, auxiliary-loss, checkpoint, and completion path passed |
| V3 event-reward state reference, 3 seeds | `learned_recovery_ppo_v8_event_reward.json` | `1139380` | `1139381` / `1139382` | Complete | Held-out forced intervention: 55.34% raw, 55.21% safe, 1.56% violations; nominal: 0% raw/safe |
| V3 nominal-only state control, 3 seeds | `learned_recovery_ppo_v9_event_reward_nominal.json` | `1139468` | `1139469` / `1139470` | Complete | Negative control-solvability result: 145/768 nominal raw (18.88%), 131/768 safe (17.06%), and 2.08% violations after 100M requested transitions/seed. Safe success is highly heterogeneous at 0%, 4.30%, and 46.88%; all seeds are retained. This is not the adaptive V5 reference and cannot serve as a reliable nominal teacher |
| V3 event-reward learned-progress visual control, 3 seeds | `visual_recovery_progress_dagger_v6_event_reward.json` | `1139383` | `1139384` / `1139553` | Complete; original aggregate `1139385` rejected correct floor-aligned budgets before verifier repair | 748/768 nominal raw success (97.40%); 708/768 forced-sweeper raw success (92.19%), but only 125 episodes produced actual unavailability and only five removed the first goal; fixed teacher, 1.92M DAgger transitions and 39,993,344 PPO steps/seed |
| Strict physical-removal calibration | `visual_recovery_strict_removal_calibration_v2.json` | — | `1139563` / `1139567` | Complete | 2 N × 12 steps rejected at 54/64 actual removals; disjoint 6 N × 24-step calibration passed 64/64. Calibration seeds are excluded from final reporting |
| Strict clean visual physical-removal evaluation, 3 seeds | `visual_recovery_strict_removal_eval_v1.json` | — | cancelled protocol-label attempts `1139568`; final `1139574` | Complete | Untouched seed base 82,000,000; 768/768 actual removals, 404 raw successes (52.60%), 402 safe (52.34%), five violations (0.65%) |
| Strict state physical-removal evaluation, 3 seeds | `visual_recovery_strict_removal_eval_v1.json` | — | cancelled protocol-label attempt `1139569`; provenance-bound replacement `1139580` | Complete | 768/768 actual removals; 23 raw successes (2.99%), 22 safe (2.86%), 272 violations (35.42%); checkpoint byte and canonical-task hashes recorded |
| Post-audit strict-removal state baseline, 3 seeds | `learned_recovery_ppo_v11_strict_removal.json` | `1139595` | strict `1139597`; nominal `1139646`/`1139647` | Complete | 99,942,400 transitions/seed; unseen strict evaluation: 756/768 raw and safe (98.44%), zero violations; first/second-removal safe 98.66%/98.22%. Complementary nominal evaluation: 0/768 raw/safe and 565/768 violations (73.57%), so this is a condition-specialized ceiling, not an integrated policy |
| Strict-state checkpoint integrity gate | `learned_recovery_ppo_v11_strict_removal.json` | — | `1139633` | Complete | All three exact-budget tasks and all 17 model + 51 optimizer floating tensors/checkpoint finite; immutable best/latest hashes recorded |
| Strict state-training effect table/figure | `strict_removal_state_training_comparison_v1.json` | — | aggregate `1139635`; figure/table `1139636` | Complete | Clean visual trails matched strict-trained state by 46.09 safe-success points [−58.33, −36.20] on identical seeds; historical-state reversal is identified as distribution shift |
| Matched integrated-mixture state upper baseline, 3 seeds | `learned_recovery_ppo_v12_integrated_mixture.json` | train `1139751`; audit `1139752` | failed pre-eval strict array `1139753`; replacement `1140056`; nominal/configured `1139754`/`1139755`; state-only strict aggregate `1139845`; gate `1139804` | Complete; teacher gate failed nominal retention | All three seeds completed 99,942,400 floor-aligned transitions with exit zero and passed the finite checkpoint audit. Frozen strict safe success is 748/768 (97.40%) with 15 violations (1.95%); first/second-removal branches are 98.66%/96.19%. Nominal raw and safe success are both 0/768 with zero violations. Gate `1139804` therefore failed only its >=90% nominal-safe check. Original strict job `1139753` emitted no result because its submitted variable alias did not match the spooled wrapper; replacement `1140056` held config/checkpoints/seeds/episodes/evaluator fixed and completed 3/3 exit zero. Config SHA-256 `d31b9985a5249c1f157a93974e666ae542e94fe0ce6eec889998b36991b70650` |
| Failure-only strict-initialized integrated state continuation, 3 seeds | `state_fallback_release_gate_v1.json`; `learned_recovery_ppo_v13_integrated_from_strict_smoke.json`; `learned_recovery_ppo_v13_integrated_from_strict.json` | fail-closed router `1139959`; smoke `1139850`; train `1139851`; audit `1139852` | strict/eval `1139853`/`1139854`; aggregates `1139855`/`1139856`; teacher gate `1139857` | Complete; gate failed: 92.58% strict safe, 91.18%/93.91% branch safe, 0.91% strict violations, but 0/768 nominal | All three exact 99,942,400-step checkpoints passed the finite audit. Despite 20% strict / 80% nominal reverse-curriculum training, catastrophic nominal forgetting persisted. The nonzero failing gate artifact released only the separately disclosed dual-specialist router; this policy is not an integrated teacher |
| Failure-only reverse-teacher RGB pair, 3 seeds | `visual_recovery_reverse_teacher_dagger_v17.json`; `visual_recovery_vicreg_reverse_teacher_smoke.json`; `visual_recovery_vicreg_reverse_teacher_v18.json` | V17 train/audit `1139860`/`1139861`; V18 smoke/train/audit `1139862`--`1139864` | strict `1139865`/`1139866`; nominal `1139867`--`1139870`; strict table `1139871`; selector `1139872`; pose `1139873`--`1139877`; task-semantic `1139878`--`1139882`; figure `1139883` | Fallback state-teacher gate failed nominal retention; zero V17/V18 allocation | The reverse-curriculum state candidate was 92.58% strict safe but 0% nominal, so it is ineligible as a single integrated teacher. All dependent jobs remain unallocated; the separately routed dual-specialist path is the only released fallback |
| Failure-only dual-specialist RGB V19, 3 seeds | `dual_specialist_teacher_gate_v1.json`; `dual_specialist_release_gate_v1.json`; `visual_recovery_dual_specialist_smoke.json`; `visual_recovery_dual_specialist_dagger_v19.json`; `strict_removal_dual_specialist_extension_v8.json`; `integrated_visual_selection_v6.json` | teacher gate `1139917`; release router `1139957`; smoke/train/audit `1139901`--`1139903` | strict/nominal `1139904`--`1139906`; strict table `1139907`; selector `1139908`; pose `1139909`--`1139911`; task-semantic `1139912`--`1139914`; figure `1139915` | Exact three-seed training/audit and held-out selector complete; eligible and selected | All seeds completed exactly 99,999,744 transitions and audit `1139903` verified finite best/latest model and optimizer tensors, exact counters, restricted observation contract, and identical source provenance. Selected steps are 26.21M, 96.66M, and 25.39M; no seed was filtered. Across 768 held-out episodes, strict raw/safe success is 750/740 (97.66%/96.35%) with 10 violations (1.30%); first-/second-removed safe success is 363/374 (97.06%) and 377/394 (95.69%). Nominal raw/safe success is 727/702 (94.66%/91.41%) with 28 violations (3.65%). Selector `1139908` passed every frozen threshold and selected V19 with 91.41% worst-endpoint safe success, releasing confirmation gate `1140359`. DAgger uses V6 (the clean learned-progress DAgger RGB controller) restricted-RGB and V11 (the strict-trained state PPO specialist) strict-state specialists through a training-only physical-resolution routing label; deployment remains restricted RGB. Privileged teachers, labels, and critic prohibit a pure-SSL claim |
| Failure-only VICReg dual-specialist RGB V20/V21 | `visual_recovery_vicreg_dual_specialist_smoke.json`; `visual_recovery_vicreg_dual_specialist_v20.json`; V21 `visual_recovery_vicreg_dual_specialist_v21_low_variance*.json`; strict/selection/representation extensions | V20 smoke/train/audit `1139931`--`1139933`; V21 one-seed smoke/gate `1140356`/`1140357`; V21 train/audit `1140381`/`1140382` | V20 strict-through-figure `1139934`--`1139945`; repair reports `1140887`/`1140888`; V21 strict/nominal/aggregate/selector `1140383`--`1140387`; matched pose/task-semantic jobs begin at `1140388` | V20 and V21 exact audit/held-out complete and rejected | V20's stronger VICReg reaches 85.42% strict and 74.06% first-removal safe success. V21 changes only variance from 0.01 to 0.001 and completed three exact 99,999,744-transition seeds. It improves nominal safe success to 92.19% but reaches only 87.63% strict and 78.34% first-removal safe success, failing the frozen integrated gate; selector `1140387` retains V19. Thus neither anti-collapse coefficient improves integrated control. Representation decodability remains diagnostic rather than causal policy evidence |
| DrAC-style dual-specialist stability V22/V23 | V22 runtime/smoke/full configs; `drac_stability_smoke_gate_v1.json`; failure-only `visual_recovery_dual_specialist_drac_v23_calibrated_runtime_smoke.json` | V22 runtime `1140573`; stopped smoke/gate `1140574`/`1140575`; unallocated full/audit `1140576`/`1140577`; failure router `1140598`; V23 runtime `1140596` | V22 strict/nominal/aggregates `1140579`--`1140582` unallocated | V22 rejected; V23 runtime complete with no larger allocation | V22's weak-BC runtime collapsed from 93.75% end success to 0%; its full-DAgger smoke produced KL 1.15e8 rising to 1.86e20 and two 0% evaluations before being stopped at 1.64M. Router `1140598` suppressed every V22 full/eval job. V23 changed only coefficient 0.1 to 0.00009 and completed exactly 262,144 steps: end success 93.75% to 92.19%, violations 3.13% to 1.56%, final score 0.9064. This prevents weak-BC collapse, but raw KL remained 1,000--1,400 and V22 showed the full-DAgger scale is many orders larger, so V23 receives no larger allocation |
| Bounded shift-action consistency V24 | runtime/smoke/full configs; runtime/stability gates; separate bounded-loss helper/trainer/wrapper; strict extension and selector V9 | failure router `1140598`; runtime/gate `1140599`/`1140609`; 20M smoke/gate `1140610`/`1140623`; gated full/audit `1140624`/`1140629` | strict/nominal `1140630`/`1140631`; aggregates/selector `1140632`--`1140634` | Rejected; all full/held-out work unallocated | V24 completed exactly 19,996,672 transitions with maximum bounded loss 0.21873. Best end success was 71.48% with 3.13% violations; score margin versus matched V19 was -27.35 points. Its last-three violation mean was 17.58% and tail score difference was -47.27 points. Gate `1140623` passed bounded finiteness and best safety but failed best success, best margin, tail safety, and tail improvement. Full `1140624` and held-out `1140629`--`1140634` remain dependency-failed. Runtime/smoke metrics are disclosed allocation diagnostics, not held-out claims |
| Mechanically scaled bounded consistency V25 | smoke/full configs; scaled stability gate; explicit-rejection router/checker; strict extension V13 (the stabilized integrated RGB controller); selector V10; causal/OOD V2; guarded allocation DAG | route `8d5c66aa...e910`; smoke `1140789`; gate `1140790`; suppressed full/audit `1140791`/`1140792` | strict/nominal/aggregates/selector/causal `1140793`--`1140799` suppressed | Rejected; exact smoke complete, all larger work unallocated | Disclosed post-hoc failure-only fallback. V25 completed exactly 19,996,672 transitions with finite bounded consistency (maximum logged loss 0.243996). It passed five of six frozen checks: 91.02% best end success, 1.95% best violations, 3.00% tail violations, +9.24-point tail-score improvement, and finite bounded loss. Its best score was 0.87124 versus V19's 0.92594: margin -5.47 points, narrowly below the required -5-point floor. Gate `1140790` therefore exited nonzero and job `1140791` is `DependencyNeverSatisfied`; the remaining full, held-out, selector, and causal/OOD chain is transitively held. Stable smoke behavior is allocation evidence only, not held-out performance or robustness evidence |
| Direct V19 continuation-stage temporal-SSL ablation V26 | `visual_recovery_dual_specialist_no_temporal_v26.json`; strict comparison V15; `temporal_ssl_continuation_ablation_v1.json` | train/audit `1140929`/`1140930` | strict/nominal `1140931`/`1140932`; aggregates/report `1140933`--`1140935`; paired verdict `1140947` | Complete; continuation-stage temporal hypothesis not confirmed | Exact V19 control except temporal coefficient 0.01→0.0 and experiment identity/claim text. V26 reaches 90.49% nominal and 93.88% strict safe success. V19's gain at V26's limiting nominal endpoint is +0.91 points with paired hierarchical 95% interval [-4.04, 6.51], below the frozen >=3-point and positive-lower-bound rule. First-/second-removal differences are +6.15/-1.02 points and also have intervals crossing zero. Both arms inherit upstream temporal training and privileged supervision, so only continuation-stage SSL is isolated; this is not a fully SSL-free lineage result |
| Selected-policy causal and renderer-native OOD evaluation | `selected_visual_causal_ood_v1.json`; `v19_incumbent_causal_ood_v1.json`; `LearnedRecovery-v3-OOD`; causal/OOD runner and aggregate | rendering preflight `1140480`; V19 incumbent `1140989`; corrected final selector `1140991` | incumbent aggregate `1140990`; final aggregate `1140992` | V19 incumbent complete; causal-head hypothesis confirmed, visual-OOD robustness rejected; final suite dependency-held on V21 | All 33 incumbent tasks and aggregate completed with exit zero (16,896 paired episodes). Cyclic progress shift reduces intervention safe success 14.32 points [0.65, 29.69], confirming the frozen causal-utility rule. OOD robustness fails: intervention safe success is 5.08% for 4-pixel shift, 2.86% camera-high, 28.91% camera-left, 42.84% dim light, 56.77% warm light, 60.16% brightness, and 69.66% warm color. The result is post-selection and simulation-only. Dry-run repair added only the missing baseline; malformed originals were canceled at zero runtime. Corrected final jobs `1140991`/`1140992` remain held on V21 and cannot be inferred from the incumbent result |
| V19 generic robust self-distillation V27 (the generic robust self-distillation candidate) | smoke/full configs; broad random shift/brightness/channel augmentation; frozen V19-action retention; privileged progress labels; development OOD and matched gate | smoke `1141054`; development `1141055`; aggregate `1141056`; original/corrected gates `1141057`/`1141079`; suppressed full `1141058` | no strict, unseen-OOD, or final evaluation allocated | Rejected by corrected matched-seed development gate | Smoke completed 2,000 updates / 512,000 transitions. It retained 85.94% nominal and 87.89% intervention safe success and preserved a 50-point causal progress drop [43.36, 57.03]. Mean OOD improvement over matched V19 seed 1788 was only +4.69 points (threshold +20), worst OOD remained 0% (threshold 25%), and camera-left intervention regressed 20.70 points (maximum allowed regression 5%). Initial checker used pooled V19 rates; corrected source reads immutable seed-1788 records, has a regression test, and rejects the same fixed candidate without changing thresholds. Generic image augmentation is development evidence only and does not solve rendered viewpoint shift |
| V19 paired rendered-domain distillation V28 (the paired rendered-domain distillation candidate) | 20-step exactly paired nominal/rendered segments; camera-left/high and dim/warm training profiles; frozen V19 action retention; privileged progress and latent consistency; explicit non-PPO accounting | failed invariant smoke `1141101`; protocol-debug `1141103`; valid smoke `1141113`; audit `1141135`; development `1141136`; aggregate/gate `1141137`/`1141138`; suppressed full branch `1141139`--`1141148` | no full standard, strict, unseen, or final result allocated | Rejected by frozen development gate | Valid smoke completed 800 updates / 102,400 student / 204,800 simulator transitions with zero paired-state error, source/render action MSE 0.00141/0.01124. It improved matched-seed mean OOD safe success by 31.14 points, retained 90.62% intervention safe success, and preserved a 29.69-point causal drop [23.05, 36.33]. It failed nominal retention (82.81% vs 85%) and worst OOD (0% vs 25%): exact pixel shift was 0%/12.89% nominal/intervention and camera-left 16.80%/45.31%. Lighting/color improved, but Boolean allocation correctly failed. Not full-episode distillation, RL, pure SSL, or real-robot evidence |
| V19 (dual-specialist RGB base) frozen-head multidomain encoder distillation V29  | V28 paired rendered segments plus exact observed pixel/brightness/color transforms; all V19 non-encoder parameters frozen; exact teacher-feature/action anchors; privileged progress labels | pre-metric failure `1141245`; corrected smoke `1141246`; audit `1141248`; development `1141249`; aggregate/gate `1141250`/`1141251` | no three-seed standard/strict/unseen allocation | Rejected by frozen gate | Encoder anchoring repaired nominal safe success to 89.45% and improved matched mean OOD 29.69 points, but intervention retention fell to 72.27% and worst OOD was 4.30%. Pixel shift reached only 4.30%/26.95% nominal/intervention; camera-left 7.81%/34.38%. Causal drop remained 18.36 points [10.55, 26.17]. Gate failed two mandatory checks; prepared final configs were never submitted. Not RL, pure SSL, full-episode distillation, or real-robot evidence |
| V19 (dual-specialist RGB base) full-episode multidomain DAgger V30  | nominal V19 action teacher; rendered nominal/strict state PPO teachers routed by physical resolution; five complete autoresetting domains; observed sensor shifts; privileged progress labels | smoke `1141281`; audit `1141283`; development `1141284`; aggregate/gate `1141285`/`1141286` | no full standard/strict/unseen allocation | Rejected by frozen gate | Completed 320,000 finite, audited full-episode DAgger transitions, but state-teacher actions overwrote V19: nominal/intervention safe success 28.91%/50.78%, mean OOD change -1.79 points, worst OOD 0%, worst individual regression 25.39 points. Only causal progress utility passed (35.55-point drop [29.30, 41.80]); gate failed six checks. Not PPO, pure SSL, from-scratch visual RL, or real-robot evidence |
| V19 (dual-specialist RGB base) same-physics multicamera DAgger V31  | one V3 physics state with simultaneous nominal/left/high cameras; frozen V19 sole action teacher; full episodes; cycled observed sensor shifts; per-view progress labels | smoke `1141316`; immutable audit; development `1141318`; aggregate/gate `1141319`/`1141320` | no three-seed standard/strict/unseen allocation | Rejected by frozen gate | Exact 256,000-transition smoke was finite and audited. It retained 90.63% nominal safe success, improved matched mean OOD by 26.20 points, preserved causal utility (21.09-point drop [12.50, 29.69]), and had no >2.34-point OOD regression. Intervention retention was only 74.22% and worst OOD was 4.69%, so the gate failed two mandatory checks. Not PPO, pure SSL, or real-robot evidence |
| V19-preserving geometry-routed RGB adapter V32 | frozen complete V19 base path; learned RGB-only domain router; separate robust encoder; same-physics cameras; observed sensor shifts; multi-view consistency and training-only 3D geometry target | pre-metric trainer/agent/evaluator/config/gate frozen | three-seed standard/strict and inherited unseen suite only after gate | Local syntax/contracts pass; Jarvis validation next | One seed-1788 smoke uses 384,000 full-episode transitions. The evaluator provides no domain label and deployment remains restricted RGB/proprioception/TCP/instruction/learned progress. Exact V19 behavior is structurally available on the learned base route. Privileged geometry, V19 targets, and observed domains prohibit pure-SSL, PPO, or held-out claims |
| V19-preserving geometry-routed RGB adapter V32 | frozen complete V19 base path; learned RGB-only domain router; separate robust encoder; same-physics cameras; observed sensor shifts; multi-view consistency and training-only 3D geometry target | stopped pre-metric scaling attempt `1141353`; corrected smoke `1141354`; immutable audit; development `1141364`; aggregate/gate `1141365`/`1141366` | no full standard/strict/unseen allocation | Rejected by frozen gate | Corrected exact 384,000-transition checkpoint is finite. It restored 94.14% nominal and 92.97% intervention safe success, preserved causal utility, and had no matched regression, passing five checks. Mean OOD gain was +17.44 points versus +20 required and worst OOD was 6.64% versus 25%; no full allocation. Geometry labels prohibit pure-SSL claims |
| V32 pixel-coordinate canonicalization diagnostic | deterministic inverse of the known observed right-four shift; exact V32 checkpoint and paired seeds | nominal/intervention `1142333`/`1142334` | none | Positive mechanism upper bound; ineligible as a method | Safe success rises from 6.64%/32.81% to 94.14%/93.75%. This isolates coordinate alignment but hard-codes the known perturbation, so it is excluded from candidate tables and gates |
| Paired learned canonical-view RGB adapter V33 | residual U-Net maps same-state camera/sensor domains to nominal RGB; direct-pixel learned router; frozen complete V19 control path | train `1142351`; audit direct; development `1142440`; aggregate/gate direct after queued CPU jobs were cancelled | no full standard/strict/unseen allocation | Rejected by frozen gate | Exact 384,000-transition checkpoint is finite. V33 retains 94.14% nominal/intervention safe success, causal utility, and +20.09-point mean OOD improvement, but worst OOD is 0% and worst domain change is -26.17 points. Five of seven checks pass. Paired images and V19 supervision prohibit a pure-SSL claim |
| V33 forced canonical-route diagnostic | exact V33 checkpoint; route threshold forced so every pixel-shift frame uses the learned canonicalizer | nominal/intervention `1142466` | none | Positive diagnosis; ineligible as a method | Pixel-shift safe success changes from 0%/11.33% under learned routing to 4.30%/37.89% under 100% canonical routing. Routing matters, but synthesis remains far below eligibility; files are explicitly suffixed and excluded from candidate claims |
| V34 (the dense spatial-warp candidate) factorized spatial/photometric canonicalization | learned dense flow warp plus bounded RGB residual; eight-way learned pixel router; exact frozen V19 nominal path; paired camera, sensor, and synchronized renderer-light domains | repaired train `1142519`; immutable audit direct; development array `1142612`; aggregate/gate direct | no multi-seed standard/strict or D-168 unseen allocation | Rejected by frozen gate after passing 6/7 checks | Exact 384,000-primary/1,536,000-total checkpoint is finite. Nominal/intervention safe success is 92.97%/94.53%; causal drop is 28.12 points [21.88, 34.38]; mean OOD gain is +45.98 points and no-large-regression passes. Worst OOD is only 1.56% because four-pixel translation reaches 1.56%/11.33%, failing the 25% allocation floor. Camera-left improves to 56.25%/83.20%; camera-high 78.12%/83.98%; dim/warm lighting 88.28%--92.19%; sensor color 85.16%--97.66%. Jobs `1142485`/`1142516` remain archived pre-metric synchronization failures. Training privilege remains disclosed |
| V35 (the observed-domain canonicalization controller) supervised translation repair on frozen V34 | RGB shift classifier plus continuous two-axis offset regressor; hard learned route; differentiable inverse warp; frozen V34 downstream | smoke `1143179`; full V34/V35/audits `1143214`--`1143217`; repaired confirmation `1143639`--`1143643` | standard, strict, D-176, three aggregates, final gate complete | Rejected for general release; final gate passes 4/10 | Observed smoke passed 7/7, but three-seed standard nominal/intervention is 81.25%/89.19% with a 73.83% minimum seed. Strict retention passes at 91.54% pooled, 82.81% minimum seed, and -4.82 points versus V19. D-176 mean unseen safe success is 18.34%, worst pooled cell 2.08%, minimum seed/domain 0%; causal progress remains positive but unseen robustness fails. V19 remains incumbent. Supervised labels and inherited V34/V19 privilege prohibit pure-SSL and end-to-end-RL claims |
| Method-information and interaction contract | method-information contract v1 | — | superseded builds `1139885`/`1139886`/`1139916`; V20-complete replacement `1139946` | Complete | JSON/CSV/Markdown enumerate 20 methods with deployment actor inputs, training-only privilege, temporal/VICReg coefficients, seeds, exact floor-aligned new PPO and DAgger interactions, and initializer/teacher checkpoint lineage. Upstream training is disclosed separately instead of being silently folded into unverifiable totals; the artifact contains no outcome metric. Builder/config hashes are frozen above |
| Matched strict/nominal benchmark table and figure | strict-removal comparison v14; `integrated_regime_comparison_v2.json` | strict aggregate `1140913` | report `1140914` | Complete three-seed screen | Source-validated JSON/CSV/Markdown/PNG/PDF compare clean V6, V13, V19, V20, and three state curricula on identical 768-episode strict and nominal protocols. V19 is the only method with a >90% worst endpoint (91.41%); V13 is 83.69%, V20 74.06%, clean V6 29.14%, and all state cohorts 0% because nominal safe success is zero. Visual/state deployment and training privilege remain explicit; wide three-seed hierarchical intervals motivate the active five-seed report |
| Matched interaction-cost accounting | `build_integrated_sample_efficiency.py`; matched outcome and method-information artifacts | failed schema assumption `1140918`; corrected build `1140927` | `integrated_sample_efficiency_v1.{json,csv,md}` | Complete | V19 consumes 99,999,744 PPO plus 1,920,000 DAgger = 101,919,744 new interactions/seed; V13 uses 99,999,744 and state baselines 99,942,400. The report validates seed counts and arithmetic, records source hashes and actor/training privilege, and discloses upstream initializer/teacher training separately. It deliberately does not invent success-per-interaction scalar scores |
| Post-audit strict-removal visual continuation, 3 seeds | `visual_recovery_strict_adaptive_v12_event_reward.json` | `1139596` | `1139598` | Cancelled at 0.8--1.1M steps after all seeds fell from 43.8--53.1% initialization success to 0% | Removed-object pose-auxiliary losses exploded to roughly 650--1,135; failed extension retained for audit |
| Stabilized integrated visual continuation, 3 seeds | `visual_recovery_strict_adaptive_v13_stable.json` | failed pre-step attempt `1139610`; corrected `1139624`; audit `1139634` | strict `1139625`; nominal `1139626` / `1139627`; selector `1139667` | Complete; all exact 99,999,744-step checkpoints finite; selector ineligible | Nominal raw/safe 92.97%/90.76%, violations 2.47%; strict raw/safe 90.89%/89.71%, violations 1.17%; first-/second-removal safe 83.69%/95.43%. Frozen thresholds were 90% strict/nominal, 85% each branch, and <=5% violations: strict and first-removal miss by 0.29/1.31 points, so five-seed gate `1139772` failed and allocated nothing. Strongest integrated restricted-RGB screen so far, but not eligible |
| Strict-state-teacher DAgger visual extension, 3 seeds | `visual_recovery_strict_teacher_dagger_v14.json` | teacher nominal eval/aggregate `1139646`/`1139647`; gate `1139651`; gated train `1139652` | planned audit/evaluation/probe chain `1139653`--`1139660` | Rejected before training; downstream chain cancelled | The strict teacher scored 0/768 nominal raw/safe with 73.57% violations, failing every >=70% raw/safe and <=5% violation allocation check. Frozen V14 config and gate artifact are retained as negative evidence; no DAgger or PPO training allocation ran. Router `1139662` released the temporal extension |
| Matched integrated-teacher RGB extension, 3 seeds | state-only strict aggregate `strict_removal_integrated_state_teacher_gate_v1.json`; `integrated_state_teacher_gate_v1.json`; `visual_recovery_integrated_teacher_dagger_v15.json` | state-only aggregate `1139845`; teacher gate `1139804`; gated train `1139805`; audit `1139806` | strict/nominal `1139807`/`1139808`; nominal aggregate `1139809`; strict comparison `1139810`; selector `1139811`; representation `1139812`--`1139814`; figure `1139815` | Gate failed; zero V15 allocation | V12 (the integrated-mixture state PPO) passed strict, both removal branches, and both violation checks but scored 0% nominal safe success against the frozen >=90% requirement. Gate `1139804` failed closed, so dependent V15/V16 primary-teacher branches consumed no GPU allocation. V15 config SHA-256 `13d8aa1e6ca2fde85f882a1d48fb93bf67bf7c882de504a090c75fa4fdc3b19e`; privileged training remains disclosed and cannot change V1--V5 |
| Anti-collapse VICReg RGB extension, 3 seeds | `visual_recovery_vicreg_smoke.json`; `visual_recovery_vicreg_integrated_teacher_v16.json` | gate-shared smoke `1139819`; gated train `1139820`; audit `1139821` | strict/nominal `1139822`/`1139823`; nominal aggregate `1139824`; strict comparison `1139825`; selector `1139826`; representation `1139827`--`1139829`; figure `1139830` | Primary teacher gate failed; zero V16 allocation | Exact V15 task/data/seed/budget ablation except self-supervised variance 0.01 and covariance 0.001 penalties and distinct trainer provenance. Because V12 failed nominal retention, neither member of this primary-teacher pair allocated. The separately gated reverse-teacher and dual-specialist paths remain the eligible post-hoc tests. Config SHA-256 `538ba61ce7f4172ac8eb3d1513ae5990eb906c342226a8b04bcf8985427682cd` |
| Task-semantic matched-pixel probes | V13/V15/V16 training configs; `visual_task_representation_vicreg_ablation_v1.json` | — | V13 probe/aggregate `1139831`/`1139832`; V15 `1139833`/`1139834`; V16 `1139835`/`1139836`; paired V16--V15 comparison `1139837` | Dependency-scheduled after immutable checkpoint audits | Predicts both ordered goal-resolution bits from exactly matched RGB datasets and reports balanced accuracy, ROC AUC, and R² against a matched random encoder. This is separate from the frozen pose probe. Because all candidates also receive training-only progress labels, the result is task-semantic decodability evidence and not a pure self-supervised or causal-control claim |
| Integrated visual-policy eligibility/selection | `integrated_visual_selection_v1.json`; `integrated_visual_selection_v2.json`; extensions `integrated_visual_selection_v3.json`/`v4.json` | — | V13-only `1139667`; V13/V14 `1139668` cancelled after V14 gate rejection; V13/V15 `1139811`; V13/V15/V16 `1139826` | Threshold rule frozen before each candidate's held-out result; dependency-scheduled | Requires >=90% strict and nominal safe success, >=85% safe success on each removed-goal branch, and <=5% violations in each condition. Eligible candidates rank by their worst strict/nominal/branch safe endpoint; no candidate is selected if all fail |
| V13 (stabilized integrated RGB base) checkpoint integrity gate  | `visual_recovery_strict_adaptive_v13_stable.json` | — | `1139634` | Dependency-queued before strict, nominal, and probe arrays | Requires exact completed budget/task/observation contract, source provenance, finite best/latest models and latest optimizer, and content hashes for all three seeds |
| Stabilized V13 matched-pixel representation probe | `visual_recovery_strict_adaptive_v13_stable.json` | — | probe `1139628`; aggregate `1139629` | Complete | Pose learned-minus-random R² −0.003 [−0.247, 0.203], so no pose-representation advantage. Separate ordered goal-resolution probe `1139831`/`1139832` is positive on every seed: balanced-accuracy +0.044 [0.040, 0.050], ROC-AUC +0.019 [0.007, 0.029], R² +0.228 [0.131, 0.344]. Supervised progress labels prohibit a pure-SSL or causal claim |
| Clean-to-V13 matched-pixel representation comparison | original `visual_representation_strict_stability_v2.json`; corrected protocol `visual_representation_probe_strict_matched_v1.json`; comparator `visual_representation_strict_stability_matched_v3.json` | CPU/monolithic/retired attempts `1140243`/`1140263`/`1140368`; item-isolated suite `1140374`; composite repair/delta `1140446`/`1140447` | probes `1140369`/`1140370`; aggregates `1140371`/`1140372`; comparator `1140373` | Complete with preserved failed suite and source-identified repair | The original comparator correctly failed on different pixels. The corrected run verified byte-identical datasets and behavior checkpoints within all three seeds. V13-minus-V6 mean learned-feature R² is +0.0330 with paired seed-bootstrap interval [0.0141, 0.0619], but V6/V13 learned mean R² are -0.0399/-0.00686 and neither reliably beats random. Claim only relative diagnostic decodability. Full isolated suite `1140374` ran 70 files/443 tests with one stale D-066 mechanism assertion; repair `1140446` reused 69 byte-identical results and reran the corrected file 2/2, then delta `1140447` passed 30/30 affected tests |
| V3 direct RGB factorial, 3 methods x 3 seeds | `visual_recovery_ppo_gate_v2_event_reward.json` | `1139393`; integrity audit `1140198` | `1139394` / `1139395` | Complete: nine policy-seed tasks, 18 endpoint files, and aggregate exited zero; audit verified nine exact 39,993,344-step finite checkpoint/optimizer pairs and provenance | Nominal raw/safe: direct 0.26%/0.26%, asymmetric 0%/0%, asymmetric+temporal 0%/0%. Configured-intervention safe: 42.71%, 43.23%, 45.05%. This rejects primary V1--V3; the strong DAgger fallback cannot rewrite them |
| V3 DAgger factorial, 2 methods x 3 seeds | `visual_recovery_dagger_ablation_v7_event_reward.json` | `1139390` | `1139391` / `1139392` | Complete | Nominal: 79.56% no-SSL vs 97.27% temporal; forced-sweeper: 80.73% vs 91.67%. Pooled temporal gains are +17.71/+10.94 points, but paired hierarchical intervals include zero |
| V3 learned-progress adaptive recovery, 3 seeds | `visual_recovery_progress_intervention_v7_event_reward.json` | `1139555` (replacement for untouched held `1139471`); integrity audit `1140176` | `1139472` / `1139473` | Complete: all seeds reached exactly 99,999,744 transitions; audit verified all three finite checkpoint/optimizer pairs and provenance | Nominal raw/safe 95.57%/94.14%; configured-intervention raw/safe 89.71%/88.02%, with 2.21% violations. These are separate endpoints and not strict-removal results |
| Matched strict evaluation of preregistered adaptive recovery | `visual_recovery_strict_removal_eval_v1.json` | — | `1139585`; aggregate `1139586` | Complete: 768/768 episodes passed actual-removal invariant | Strict raw/safe 33.20%/32.42%, 1.69% violations; first-/second-goal-removed safe 20.59%/43.65%. Clean-minus-adaptive strict safe is +19.92 points with paired hierarchical 95% interval [−0.13, 48.83], so V4 is not confirmed |
| V3 automatic held-out competence gate | `visual_recovery_progress_dagger_v6_event_reward.json` | — | `1139554` | Passed: 97.40% nominal over 768 episodes | Original `1139500` inherited the failed aggregate dependency; replacement gate passed but Slurm's administrative hold prevented user release, so identical array `1139555` was submitted directly from the passed gate evidence |
| Strict V1--V5 hypothesis report | `visual_recovery_hypothesis_validation_v1.json` | — | retired dependency-dead `1139529`; replacement `1139614` | Complete: V1 rejected, V2 rejected, V3 rejected, V4 rejected, preregistered-primary V5 confirmed | V5 compares V7 (the adaptive continuation controller) against the originally frozen 2.86%-safe historical state reference and cannot support competitiveness against the later distribution-matched 98.44%-safe strict state PPO; generated JSON/Markdown preserve both the verdict and claim boundary |
| Primary cross-method visual/state table | `visual_recovery_comparison_v2_event_reward.json` | — | retired dependency-dead `1139530`; replacement `1139615` | Nominal and forced-sweeper-condition endpoints only; required direct/adaptive inputs are fail-closed |
| Final held-out control/sweeper figure | comparison JSON from `1139615` | — | retired `1139531`; replacement `1139616` | Labels the confounded endpoint as forced-sweeper condition, never physical removal or recovery |
| Full strict-removal extension table/figure | `strict_removal_extension_comparison_v1.json` | — | replacement aggregate `1139630`; figure/table `1139631` | Five matched cohorts, actual removal required in every episode; Markdown/CSV/PNG/PDF with hierarchical uncertainty and source hashes |
| Complete clean-cohort learning curves | all three V3 clean configs | — | `1139532` | Dependency-queued after train arrays `1139383`, `1139390`, and `1139393` | Full 40M-step PNG/PDF diagnostics; individual seeds plus mean ± sample SD; explicitly non-held-out |
| V3 no-SSL DAgger adaptive recovery, 3 seeds | `visual_recovery_dagger_intervention_v8_event_reward.json` | `1139510`; integrity audit `1140175` | `1139511` / `1139512` | Complete: all seeds reached exactly 99,999,744 transitions; audit verified all three finite checkpoint/optimizer pairs and provenance; evaluations exited zero | Nominal raw/safe 86.98%/83.33% with 4.04% violations; configured-intervention raw/safe 85.55%/83.72% with 3.13% violations. Seed dispersion is large (nominal raw 98.05%, 94.53%, 68.36%), so this remains the no-temporal member of the matched V8/V9 attribution pair, not a final candidate |
| V3 temporal-SSL DAgger adaptive recovery, 3 seeds | `visual_recovery_temporal_dagger_intervention_v9_event_reward.json` | original `1139516`; corrected seed-1788 resume `1140493`; audit `1140494` | `1139517` / `1139518` | Complete and audited; held-out evaluation resource-queued | All three seeds reached exactly 99,999,744 transitions. Corrected task 2 resumed seed 1788 from 79,462,400 with model, optimizer, RNG, iteration, and counter state intact; audit `1140494` verified finite best/latest model and optimizer tensors, the observation contract, source provenance, and exact counters for every seed. Seed 1788's training-stream best was 95.31% end success with 1.56% violations, not held-out evidence. The earlier mistaken redundant resume remains disclosed, no seed is discarded, and downstream array `1139517` plus aggregate `1139518` remain fail-closed |
| V3 no-SSL / temporal adaptive representation probes | V8 / V9 configs | — | `1139514` / `1139515`; `1139520` / `1139521` | V8 complete; V9 probe priority-queued | V8 learned R² 0.661 versus random 0.339, difference +0.323 [0.210, 0.456], all seed differences positive. The audited V9 completion released all three probe tasks; aggregate `1139521` remains fail-closed on the complete array. Pose auxiliary supervision is present in both V8/V9, so only their eventual matched difference can inform the incremental temporal-loss association; neither alone supports a pure-SSL claim |
| V3 clean learned-progress representation probe, 3 seeds | `visual_recovery_progress_dagger_v6_event_reward.json` | — | `1139494` / `1139496` | Probe/seed-aware aggregate dependency-queued after clean training | 8,192 train + 8,192 held-out samples/seed; learned-versus-random pose decoding |
| V3 adaptive learned-progress representation probe, 3 seeds | `visual_recovery_progress_intervention_v7_event_reward.json` | — | `1139495` / `1139497` | Complete | Learned R² 0.725 versus random 0.339; learned-minus-random +0.387 [0.312, 0.488], all three seed differences positive. V7 also uses privileged pose auxiliary supervision and supervised progress labels, so the gain cannot be attributed to temporal SSL; diagnostic linear decodability only |
| Matched-pixel representation comparison | `visual_representation_comparison_v1.json` | — | `1139536` | Dependency-queued after all four probe aggregates | Requires identical per-seed RGB/label hashes and random-encoder controls; temporal-vs-no-SSL R2 is diagnostic, not causal control evidence |
| Obsolete forced-sweeper five-seed chain | V7 visual + historical V8 state | `1139537`--`1139539` | `1139540`--`1139547` | Cancelled before allocation; every job elapsed 00:00:00 | Retired because the old endpoint rarely produced actual removal and therefore could not confirm the corrected recovery hypothesis |
| Integrated five-seed allocation gate | `integrated_five_seed_confirmation_v2.json` | — | `1139772` | Dependency-queued after V13 selector `1139667` | Authorizes seeds 71064/84293 only if V13 passes >=90% strict and nominal safe success, >=85% in both removal branches, and <=5% violations in both regimes; exits nonzero otherwise |
| Conditional V19 five-seed training DAG | `dual_specialist_five_seed_confirmation_v1.json`; V6/V11/V13/V19 `*_confirm_append.json` configs | V19 gate `1140359`; V6/V11 arrays `1140360`/`1140361`; audits `1140362`/`1140363`; V13/audit `1140364`/`1140365`; V19/audit `1140366`/`1140367` | Held-out five-seed evaluation not yet allocated | Gate passed; V11 new-seed audit complete, V6 new-seed array running | Confirmatory seeds 71064/84293 were fixed before V19 held-out results. Gate `1140359` verified every selector threshold and released both new-seed V6 and V11 arrays. Both V11 seeds completed exactly 99,942,400 steps; audit `1140363` verified finite 17-tensor agents and 51-tensor optimizers and retained selected steps 93,323,264 and 75,300,864. V13 requires the audited V6 initializer, and V19 requires audited V6/V11 teachers plus audited V13 initialization. All append tasks are byte-equivalent to their screening configs except the seed list, and neither new seed may be discarded |
| New-seed clean V6 initializer | `visual_recovery_progress_dagger_v6_confirm_append.json` | `1139773`; audit `1139774` | — | Strict-gated, no allocation yet | Byte-identical V6 task with only seeds changed; appends new seed directories without overwriting screening checkpoints |
| New-seed integrated V13 visual policy | `visual_recovery_strict_adaptive_v13_confirm_append.json` | `1139775`; audit `1139776` | strict `1139779`; nominal `1139780`; five-seed aggregate `1139783` | Strict-gated behind clean initializer | Byte-identical V13 task; both new seeds retained regardless of outcome |
| New-seed integrated state comparator | `learned_recovery_ppo_v12_integrated_confirm_append.json` | `1139777`; audit `1139778` | strict `1139781`; intervention/nominal `1139782`; five-seed aggregate `1139784` | Strict-gated, no allocation yet | Byte-identical integrated-mixture state task; screening results are explicit dependencies of five-seed aggregation |
| Five-seed strict and dual-regime report | five-seed view configs; `strict_removal_integrated_five_seed_comparison_v4.json`; `integrated_regime_five_seed_comparison_v2.json` | — | strict `1139785`; benchmark table/figure `1139786` | Fully dependency-scheduled | Requires all five seeds and 1,280 held-out episodes in each strict/nominal condition; writes distinct artifacts and never overwrites three-seed aggregates |
| Five-seed matched-pixel representation confirmation | `visual_recovery_progress_dagger_v6_five_seed_view.json`; `visual_recovery_strict_adaptive_v13_five_seed_view.json`; `visual_representation_strict_stability_five_seed_v3.json` | — | new-seed clean/V13 probes `1139796`/`1139797`; five-seed aggregates `1139798`/`1139799`; comparison `1139800` | Fully dependency-scheduled behind the same integrated confirmation audits | Extends the fixed 8,192+8,192 matched-pixel probe to all five retained policies. Five-seed aggregates use a distinct filename, preserving the immutable three-seed screening artifacts; the paired encoder R² result remains diagnostic rather than causal control evidence |
| Selected V19 qualitative capture | V19 seed 4796, chosen by strongest joint held-out strict/nominal seed endpoint | — | branch capture `1140898`; candidate montage `1140899`; promoted hero `1140903` | Complete; metadata and sampled frames inspected | First/second actual-removal and nominal captures are safe successes from the fixed seed range, use checkpoint step 96,657,408, record the frozen selector/checkpoint/source hashes, and report zero teleport calls. First-removal uses episode seed 92,000,001; second-removal and nominal use 92,000,000. The three-panel GIF shows manipulation across all branches and was promoted to `learned-recovery-montage.gif`; SHA-256 `cacf4589...6803` |
| V3 validated recovery video branches | `visual_recovery_progress_intervention_v7_event_reward.json` | — | `1139498` | Dependency-queued after held-out recovery aggregate | Declared seed search for safe first-removed, second-removed, and nominal recordings; frozen restricted actor, zero teleport calls |
| V3 clean-policy video branches | `visual_recovery_progress_dagger_v6_event_reward.json` | — | `1139558`; retired weak-intervention search `1139561`; locked strict capture `1139587` | Complete | Both intervention panels verify actual removal and safe success under the strict config; nominal is also safe. Zero teleport calls. Strict-labeled README hero SHA-256 `c8bb83eb4db44f7e6bf40c455c2fd46bf3a7ae6236be37df5e719d1af9bfa6d8` |
| Final encoder linear probes, 3 seeds | `visual_recovery_intervention_v1.json` | — | `1139265` | Dependency queued | Analysis-only pose labels; learned vs random encoders |
| Pose-auxiliary encoder linear probes, 3 seeds | `visual_recovery_privileged_aux_intervention_v4.json` | — | `1139345` | Cancelled before allocation after reward audit | V2 probe deferred; final V3 representation will be probed instead |
| Per-seed full-task initializer selection | `visual_recovery_selection_v1.json` | — | `1139358` | Cancelled | Superseded by evidence-driven pruning of incompatible strict candidates |
| Selected visual adaptive recovery, 3 seeds | `visual_recovery_selected_intervention_v2.json` | `1139359` | `1139360` / `1139361` | Cancelled | Superseded by dedicated progress-token and pose-auxiliary branches |

## Reward-audit fork

The V2 checkpoints and logs remain preserved, but their training arrays were
cancelled after the reward audit to free GPUs for claim-eligible V3 work. Their
negative evidence is not silently discarded. They are no longer final-claim
candidates: V2's persistent
completed-goal reward makes delaying terminal success preferable at the
configured discount. The isolated `LearnedRecovery-v3` correction has a direct
zero-reward stalling regression test and does not alter the module imported by
V2 jobs. Full V3 state and visual arrays were submitted only after smoke job
`1139377` passed. No V3 result was observed before this protocol change. Their
immutable dependency chains are `1139380 -> 1139381 -> 1139382` and
`1139383 -> 1139384 -> 1139385`.

The final result table must be generated from held-out JSON artifacts, not this
ledger or training-time metrics. A job being complete does not imply a
hypothesis passed.

## 2026-08-28 live audit

- At 0.81–5.32M environment steps, the strict RGB student reached a mean of
  0.53–1.00 completed goals per episode while full two-goal success remained
  0% in all but one transient 1.56% evaluation. This is evidence for a
  sequential-transition bottleneck, not evidence of full-task success.
- At 9.76M steps, the fresh clean state reference had per-seed full-task
  success of 12.5%, 23.4%, and 4.69%. These are noisy training-time snapshots;
  only the scheduled held-out aggregate is claim eligible.
- Jobs `1139339` (pose auxiliary) and `1139347` (pixel-predicted-progress DAgger) remain
  queued behind GPU demand. Six project GPUs are currently occupied by the
  strict RGB and fresh-state controls.
- The 24-hour wrappers now requeue the same Slurm array element after the
  atomic `latest.pt` checkpoint instead of submitting a new job. This keeps
  downstream `afterok` evaluation jobs blocked across allocations. Slurm's
  signal targets the Python job process rather than using the `B:` batch-shell-
  only prefix, so the trainer's atomic-save handler actually receives
  `SIGUSR1`. The scheduler/actor contract suite passes locally and on Jarvis.
- The original symbolic-progress candidate was replaced before allocation by
  a stronger restricted-input variant. Its CNN predicts progress from RGB;
  the actor never reads `extra.goal_progress`. The updated contract suite
  passes 8/8, and full job `1139347` is gated on smoke job `1139370`.
- Before either learned-resolution job allocated, its DAgger source was frozen
  to state teacher seed 1788 for every student seed. That teacher achieved
  256/256 on the completed clean nominal evaluation; matching teachers 4796
  and 9351 achieved only 125/256 and 17/256. This expert-quality selection is
  disclosed and does not inspect any visual-policy evaluation.
  Seed 1788 is not a safety oracle: nominal safe success was 52.0% with a
  48.0% full-horizon violation rate. It supplies complete sequencing behavior;
  the student remains required to learn safety and meet the matched V3 state
  reference's violation gate during PPO and held-out evaluation. The earlier
  8.59% V2 value is retained only as historical context.
- Live `sstat` measurements showed 6.4--6.9 GiB peak resident host memory for
  the visual trainer, while its 96 GiB reservation stranded otherwise idle
  GPUs on 251 GiB nodes. Pending visual train jobs and the continuation wrapper
  now request 40 GiB, retaining over 5x measured headroom without changing any
  learning hyperparameter.
- Reducing the smoke reservation to 20 GiB and the learned-resolution branch
  to 16 GiB started the smoke and two full seeds on otherwise stranded GPUs.
  The full learned process measured 4.5--4.7 GiB RSS during startup, leaving
  more than 3x headroom; the third seed remains resource-queued. The cluster is
  running eleven project GPUs concurrently.
- Learned-resolution seeds 9351 and 4796 completed 7,500 DAgger updates each
  (1.92M transitions). Mean final-100 BC loss was 0.0528/0.0536 and
  training-distribution resolution-bit accuracy was 98.80%/98.87%. The first
  autonomous full-task evaluation was 14.06% success with 20.31% violations
  for seed 9351 and 0% success/0% violations for seed 4796. Seed 9351's
  autonomous bit accuracy fell to 79.32%, while seed 4796's 99.996% occurred
  on nearly all-unresolved episodes and is not evidence of transition
  recognition. These are initialization diagnostics, not held-out claims.
- At 0.811M PPO steps, both learned-resolution seeds had 0% training-time
  violations. Seed 9351 retained 7.81% full-task success with 0.844 mean goals;
  seed 4796 had 0% full success but improved to 0.75 mean goals. Autonomous
  resolution-bit accuracy was 91.61%/91.98%. This shows early safety recovery
  without complete loss of sequencing, but is not a hypothesis confirmation.
- The matched V3 learned-progress cohort reached 23.44%, 32.81%, and 37.50%
  full ordered two-goal success at 2.449M PPO steps, with 0/192 aggregate
  training-evaluation violations and per-seed mean completed goals of 1.08,
  1.23, and 1.28. The corresponding V2 learned-progress seeds remained at 0%
  full success near 7.5M while hovering around 0.92 completed goals. This is
  strong training-stream evidence that the event-reward correction removed the
  stalling incentive; it is not a held-out V1 confirmation.
- At the same 2.449M-step snapshot, the V3 DAgger + pose-auxiliary arm without
  temporal SSL or an explicit progress head reached 29.69%, 62.50%, and 53.12%
  (48.44% pooled) with one violation in 192 training-evaluation episodes. The
  temporal+learned-progress arm was 31.25% pooled with zero violations. This
  reverses the early candidate ranking: the event reward is supported, but an
  explicit progress head is not yet supported. The queued temporal/no-progress
  arm is required to separate the progress-head and temporal-loss effects.
- At 3.269M steps, the temporal+learned-progress arm reached 87.50%, 70.31%,
  and 64.06% full success (73.96% pooled; 2.60% pooled violations). The
  no-progress/no-SSL arm reached 21.88%, 96.88%, and 64.06% (60.94% pooled;
  1.04% pooled violations). The main candidate now leads in pooled success and
  worst-seed success, while the ablation shows extreme optimizer-seed
  sensitivity. These fixed-seed training evaluations motivate continued
  training but do not replace the scheduled held-out hierarchical comparison.
- At the predeclared 4.907M controller-retention checkpoint, the main V3 visual
  seeds reached 89.06%, 87.50%, and 90.62% full success (89.06% pooled) with
  zero violations. The kickstarting fallback is rejected as unnecessary.
  Progress-bit accuracy was only 61.85%, 47.38%, and 55.99%; high control
  success must not be rewritten as confirmation of semantic progress
  prediction. The held-out predictor audit and temporal/no-progress arm remain
  mandatory for the mechanism claim.
- The matched no-progress/no-SSL DAgger arm was 40.62%, 93.75%, and 92.19% at
  4.907M (75.52% pooled, zero violations). The main arm leads by 13.54 pooled
  points and raises worst-seed success by 48.44 points, but this contrast still
  bundles temporal SSL with the progress head. Only the queued temporal/no-
  progress arm can attribute the difference.
- The final V3 state-reference aggregate passed its provenance and protocol
  checks. On 768 held-out forced-intervention episodes it recorded 425 raw
  successes (55.34%, Wilson [51.81%, 58.82%]), 424 safe successes (55.21%,
  Wilson [51.67%, 58.69%]), 1.56% violations, and 1.010 mean completed goals.
  Hierarchical 95% intervals were [48.70%, 63.28%] raw and [48.44%, 63.41%]
  safe. Per-seed raw success was 63.67%, 50.78%, and 51.17%. The same policy
  scored 0/768 in the matched nominal condition, so it is used only as the V5
  recovery reference; job chain `1139468 -> 1139469 -> 1139470` remains the
  separately trained nominal-control upper baseline.
- Before any clean V3 held-out aggregate existed, two additional recovery
  chains were staged behind fail-closed method-specific competence gates. Jobs
  `1139510 -> 1139511 -> 1139512` test the no-SSL DAgger initializer; jobs
  `1139516 -> 1139517 -> 1139518` test its matched temporal-SSL counterpart.
  Gates `1139522` and `1139523` require the corresponding clean method in
  aggregate `1139392` to reach at least 70% across exactly three seeds and 768
  episodes before releasing any recovery GPU. Probe chains `1139514 ->
  1139515` and `1139520 -> 1139521` use the same frozen recovery checkpoints.
  This post-preregistration extension is disclosed and exists to test whether
  temporal self-supervision survives adaptive recovery without the privileged
  learned-progress actor input.
- Before the first V3 visual held-out job ran, the evaluator and aggregator
  were hardened to record evaluator/environment source hashes and fail closed
  on a wrong method, training seed, condition, environment, observation
  contract, checkpoint selector, episode count, transition budget, or
  recomputed raw/safe-success count. Paired effects now independently verify
  the intervention and instruction branches episode by episode. This changes
  no policy or evaluation trajectory; it prevents stale or misrouted JSON from
  becoming an authoritative result.
- The same pre-evaluation audit found that adaptive-policy sample accounting
  would otherwise omit the clean initializer's PPO and DAgger interactions.
  Evaluation now follows `initialization.json` back to the frozen source
  checkpoint and its `bc_pretraining.json`, records online recovery PPO,
  initializer PPO, local DAgger, and inherited DAgger separately, and verifies
  their sum. It also separates the interactions present in a selected best
  checkpoint from the full predeclared training/selection budget recorded by
  each `TRAINING_COMPLETE.json`. Thus a checkpoint selected at 10M during a
  40M run cannot be advertised as costing only 10M interactions, and a 100M-
  step adaptive run cannot be reported as though it were trained from scratch
  in only 100M interactions.
- The same pre-held-out audit replaced raw progress-bit accuracy as the sole
  learned-progress diagnostic with a complete confusion matrix, positive and
  negative recall, balanced accuracy, target prevalence, and predicted-positive
  rate. Aggregation recomputes every rate from integer counts and fails closed
  on a missing or inconsistent schema. This exposes an all-zero progress-head
  collapse that ordinary accuracy could hide; it changes evaluation only, not
  policy training or checkpoint selection.
- Before any representation probe ran, its data collection was changed from
  each evaluated policy's own rollouts to one frozen seed-matched behavior
  checkpoint. All encoders now see byte-identical RGB/label tensors within a
  seed, recorded by SHA-256 digests. The matched comparator rejects any pixel,
  label, behavior-checkpoint, seed, or random-control mismatch and treats linear
  decodability as diagnostic rather than causal control evidence. The new
  top-level config field does not enter checkpoint task dictionaries, so active
  and resumable training tasks remain byte-for-byte unchanged. Probe collection
  also resets process-global NumPy, CPU-Torch, and CUDA-Torch RNGs immediately
  before each environment reset, preventing architecture initialization from
  perturbing the matched trajectory. Ridge regularization is an explicit record
  field rather than an undocumented CLI default; the comparator also requires
  identical probe source hashes, targets, sample counts, and regularization.
- Representation-probe aggregation now rejects wrong reward semantics, method,
  training seed, observation contract, sample count, target schema, checkpoint
  lineage, source hashes, non-finite metrics, or inconsistent learned-minus-
  random arithmetic. README capture is independently blocked unless the frozen
  method's aggregate matches the V3 state raw/safe/violation thresholds and
  retains at least 70% nominal success. Only then does it select the first safe
  success from the declared sequential seed range for each branch.
- Strict report job `1139529` is dependency-queued on the four aggregate chains
  needed for the primary V1--V5 tests. It writes JSON and Markdown verdicts,
  excludes training-stream metrics, recomputes paired effects from frozen
  episode records, and labels every primary hypothesis confirmed, rejected, or
  pending. A favorable protocol extension is displayed separately and cannot
  convert a rejected primary hypothesis into a confirmation.
- Cross-method table job `1139530` has the same primary dependencies and runs
  without `--allow-missing`. Direct RGB, asymmetric, temporal, both DAgger
  clean ablations, learned-progress clean control, and primary adaptive
  recovery are mandatory. The two post-preregistration adaptive extensions
  carry explicit `required: false` markers: a failed competence gate remains a
  visible missing extension but cannot block or weaken the strict primary table.
  Its generated Markdown table uses the same JSON values, hierarchical
  intervals, and full protocol-interaction accounting as CSV/JSON, avoiding
  manual transcription into the README or generated reports.
- Figure job `1139531` accepts only V3 comparison JSON with no missing primary
  aggregate. It plots held-out nominal success and forced-intervention safe
  success with hierarchical seed/episode intervals, plus violation rates and
  the matched state-reference thresholds. It cannot read training logs or
  training-stream best metrics.
- Learning-curve job `1139532` waits for every direct, DAgger-factorial, and
  learned-progress clean training task, then renders the full 40M-step curves.
  Its title and metadata label the figure as training-stream diagnostics, not
  held-out evidence; it cannot replace jobs `1139530`/`1139531` for claims.
- Before any V3 visual held-out evaluation completed, the V3 prose was narrowed
  to match its executable verdict: only held-out success can confirm V3.
  Training-stream learning curves remain descriptive optimization diagnostics
  and cannot rescue a failed held-out comparison. No model, seed, budget,
  aggregate, or numerical acceptance threshold changed.
- Confirmatory gate job `1139545` was cancelled while dependency-pending
  after an audit found that Slurm parses comma-separated export values as
  separate environment entries. Replacement `1139546` uses the verified
  colon-delimited release list `1139537:1139538`; neither gate ran and no
  training allocation or result was affected by the replacement.
- Before any confirmatory evaluation became eligible, a seed-range audit found
  that the original linear held-out formula exceeds NumPy `RandomState`'s
  accepted range for training seeds 71064 and 84293. Screening seeds retain
  the exact existing formula. Overflow cases now use a domain-separated
  SHA-256 derivation into the signed 31-bit range; every evaluation records its
  eight batch seeds, and aggregation recomputes them and rejects missing,
  mismatched, or colliding provenance. The 16 precomputed confirmation batch
  seeds are distinct. This is a pre-execution compatibility correction, not a
  result-contingent seed change; training seeds and episode counts are
  unchanged.
- The entire replacement forced-sweeper confirmation chain `1139537`--`1139547`
  was cancelled at zero elapsed time before its gate ran. It was superseded by
  strict/nominal V13 (the stabilized integrated RGB controller) chain `1139772`--`1139786`, whose append configs are
  contract-tested byte-equivalent at the task level and whose final views
  require all five seeds. Confirmatory checkpoint audits use distinct filenames
  so they cannot overwrite the completed screening audits.
- Before V12 (the integrated-mixture state PPO) or any V15/V16 held-out result existed, the integrated state-
  teacher allocation gate was found to depend on a broad strict table that also
  waited for unrelated V7 (the adaptive continuation controller)/V13 visual cohorts. A state-only strict config and
  aggregate job `1139845` consumed the same immutable V12 strict records; gate
  `1139804` consumed that artifact plus V12's nominal aggregate and failed only
  nominal retention. The six thresholds, seeds, episode counts, candidate
  tasks, and results were unchanged. This removed an unrelated dependency and
  recorded the gate-config hash rather than changing an acceptance rule.
- The failure-only state continuation was initially drafted with V12's 80%
  strict training mixture. Before its smoke or any downstream job allocated,
  the running scratch policy plateaued near 50% on its balanced training-only
  diagnostic, consistent with one-regime specialization. Because the fallback
  initializes an already 98.44%-strict specialist, reinforcing it with 80%
  strict samples was directionally wrong. The frozen fallback now mirrors
  V13's curriculum direction: 20% strict retention and 80% nominal learning,
  with the same balanced selector, seeds, budget, held-out endpoints, and six
  acceptance thresholds. This is a disclosed post-hoc fallback design change,
  not a modification to V12 or any observed held-out result.
- During validation of the first completed nominal records, before the primary
  aggregate became eligible, the verifier was found to round configured PPO
  steps upward to a rollout batch while the trainer deliberately executes only
  complete batches by floor division. The verifier now reproduces the trainer's
  exact arithmetic (39,993,344 steps for the declared 40M configuration), with
  a non-divisible-budget regression test. All three immutable completion
  markers and held-out records already reported that correct executed budget;
  no checkpoint, episode, or result was modified.
- Aggregate `1139385` began in the short interval between held-out completion
  and the corrected verifier sync, so it failed on the now-fixed upward-rounding
  check. Replacement `1139553` consumed the same six immutable records and
  completed. Gate `1139554` then passed, but Slurm had converted the original
  user hold on `1139471` into an administrator hold after its failed dependency,
  making user release impossible. The untouched array was cancelled only after
  identical replacement `1139555` was submitted and downstream jobs `1139472`
  and `1139495` were atomically redirected to it. No training task from either
  array had started, and all later aggregate/report dependencies remain intact.
