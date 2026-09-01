#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-stage
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/peg_stage_%A_%a.out
#SBATCH --error=results/slurm/peg_stage_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
read -r -a seeds <<< "${ATR_PEG_TRAINING_SEEDS:-9351 4796 1788}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
run_root="${ATR_PEG_RUN_ROOT:?ATR_PEG_RUN_ROOT is required}"
output_dir="${ATR_PEG_DIAGNOSTIC_DIR:?ATR_PEG_DIAGNOSTIC_DIR is required}"
mkdir -p "${output_dir}"
.venv/bin/python scripts/diagnose_external_peg_checkpoint.py \
  --checkpoint "${run_root}/seed_${seed}/${ATR_PEG_CHECKPOINT_NAME:-latest.pt}" \
  --num-envs "${ATR_PEG_DIAGNOSTIC_ENVS:-128}" \
  --steps "${ATR_PEG_DIAGNOSTIC_STEPS:-100}" \
  --seed "$(( ${ATR_PEG_DIAGNOSTIC_SEED_BASE:-421710000} + SLURM_ARRAY_TASK_ID ))" \
  --output "${output_dir}/seed_${seed}.json"
