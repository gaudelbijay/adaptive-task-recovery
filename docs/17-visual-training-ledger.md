# Jarvis visual training ledger

This ledger separates exploratory jobs from experiments eligible for final
claims. Times are Jarvis scheduler times in America/New_York. All arrays use one
L40S per task, atomic checkpoints, and the repository's 24-hour continuation
wrapper.

| Stage | Config | Train job | Eval / aggregate | Status | Claim use |
|---|---|---:|---:|---|---|
| Direct visual gate, 3 methods × 3 seeds | `visual_recovery_ppo_gate_v1.json` | `1139228` (tasks 3–8), `1139237` (tasks 0–2) | `1139242` / `1139243` | Running | Exploratory: training process loaded v1 skip semantics; held-out evaluation uses clean semantics |
| First-goal visual curriculum, 3 seeds | `visual_recovery_curriculum_v1.json` | `1139246` | — | Dependency queued | Pretraining only; never reported as full-task success |
| Ordered two-goal transfer, 3 seeds | `visual_recovery_transfer_v1.json` | `1139247` | `1139255` / `1139256` | Dependency queued | Eligible under `LearnedRecovery-v2` |
| State-teacher visual bootstrap, 3 seeds | `visual_recovery_distilled_v1.json` | `1139252` | `1139253` / `1139254` | Dependency queued | Eligible, disclosed privileged training |
| Physical-intervention visual recovery, 3 seeds | `visual_recovery_intervention_v1.json` | `1139257` | `1139258` / `1139259` | Dependency queued | Primary final method if gates pass |
| Clean-semantics state PPO re-evaluation, 3 methods × 3 seeds | `learned_recovery_ppo_v6.json` | existing checkpoints | `1139262` / `1139263` | Dependency queued | Clean state baselines; original result files remain untouched |
| Final encoder linear probes, 3 seeds | `visual_recovery_intervention_v1.json` | — | `1139265` | Dependency queued | Analysis-only pose labels; learned vs random encoders |

The final result table must be generated from held-out JSON artifacts, not this
ledger or training-time metrics. A job being complete does not imply a
hypothesis passed.
