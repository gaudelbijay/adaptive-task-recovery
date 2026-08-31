#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-forward-distill
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/v4_forward_distill_%A_%a.out
#SBATCH --error=results/slurm/v4_forward_distill_%A_%a.err

set -euo pipefail
SEEDS=(9351 1788 4796)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_ROOT="${ATR_FORWARD_DISTILL_OUTPUT:-results/distillation/v4_forward_handoff}"
mkdir -p results/slurm "${OUTPUT_ROOT}/seed_${SEED}"
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/distill_v4_forward_handoff.py \
  --teacher "${ATR_FORWARD_TEACHER:-results/learned_recovery/learned_recovery_ppo_v11_strict_removal/event_reward_strict_removal_state_ppo/seed_9351/best.pt}" \
  --output "${OUTPUT_ROOT}/seed_${SEED}/distilled.pt" \
  --seed "${SEED}" \
  --handoff-step "${ATR_FORWARD_HANDOFF_STEP:-4}" \
  --episodes "${ATR_FORWARD_DISTILL_EPISODES:-16}" \
  --teacher-rollout-episodes "${ATR_FORWARD_TEACHER_EPISODES:-2}"
