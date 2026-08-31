#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-learned-router
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-14
#SBATCH --output=results/slurm/v4_learned_router_%A_%a.out
#SBATCH --error=results/slurm/v4_learned_router_%A_%a.err

set -euo pipefail
mkdir -p results/slurm "${ATR_ROUTER_EVAL_OUTPUT:-results/v4_learned_router_development}"
export PYTHONUNBUFFERED=1
FIXED_OPTION_ARGS=()
if [[ -n "${ATR_FIXED_OPTION:-}" ]]; then
  FIXED_OPTION_ARGS+=(--fixed-option "${ATR_FIXED_OPTION}" --fixed-option-start-step "${ATR_FIXED_OPTION_START_STEP:-1}")
fi
ENSEMBLE_ARGS=()
if [[ "${ATR_NOMINAL_ENSEMBLE:-0}" == "1" ]]; then
  ENSEMBLE_ARGS+=(--nominal-ensemble --nominal-ensemble-reduction "${ATR_NOMINAL_ENSEMBLE_REDUCTION:-mean}")
fi
POLICY_MEMBER_ARGS=()
if [[ -n "${ATR_NOMINAL_POLICY_INDEX:-}" ]]; then
  POLICY_MEMBER_ARGS+=(--nominal-policy-index "${ATR_NOMINAL_POLICY_INDEX}")
fi
if [[ -n "${ATR_TEMPORARY_POLICY_INDEX:-}" ]]; then
  POLICY_MEMBER_ARGS+=(--temporary-policy-index "${ATR_TEMPORARY_POLICY_INDEX}")
fi
.venv/bin/python scripts/evaluate_v4_learned_option_router.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --router-checkpoint "${ATR_ROUTER_CHECKPOINT:?set ATR_ROUTER_CHECKPOINT}" \
  --router-metadata "${ATR_ROUTER_METADATA:-results/router/v4_option_prefixes_train_v1.json}" \
  --permanent-state-checkpoint "${ATR_PERMANENT_CHECKPOINT:-results/manipulation_ppo/learned_recovery_v4_delayed_permanent_transfer/delayed_permanent_state_transfer/seed_9351/delayed_frozen_iter24.pt}" \
  --reverse-state-checkpoint "${ATR_REVERSE_CHECKPOINT:-results/learned_recovery_v4/learned_recovery_v4_reverse_state_pilot/reverse_ejection_state_specialist/seed_9351/reverse_frozen_iter424.pt}" \
  --forward-state-checkpoint "${ATR_FORWARD_CHECKPOINT:-results/learned_recovery/learned_recovery_ppo_v11_strict_removal/event_reward_strict_removal_state_ppo/seed_9351/best.pt}" \
  --output-dir "${ATR_ROUTER_EVAL_OUTPUT:-results/v4_learned_router_development}" \
  --seed-base "${ATR_ROUTER_EVAL_SEED_BASE:-310000000}" \
  --episodes "${ATR_ROUTER_EPISODES:-128}" \
  --confirmation-steps "${ATR_CONFIRMATION_STEPS:-2}" \
  --force-scale "${ATR_ROUTER_FORCE_SCALE:-1.0}" \
  --onset-step "${ATR_ROUTER_ONSET_STEP:-0}" \
  --return-delay "${ATR_ROUTER_RETURN_DELAY:-30}" \
  --control-delay "${ATR_ROUTER_CONTROL_DELAY:-0}" \
  --safe-hold-until-step "${ATR_ROUTER_SAFE_HOLD_UNTIL_STEP:-0}" \
  "${FIXED_OPTION_ARGS[@]}" \
  "${ENSEMBLE_ARGS[@]}" \
  "${POLICY_MEMBER_ARGS[@]}"
