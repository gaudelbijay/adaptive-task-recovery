#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-bc
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/peg_bc_%A_%a.out
#SBATCH --error=results/slurm/peg_bc_%A_%a.err

set -euo pipefail
mkdir -p results/slurm
read -r -a seeds <<< "${ATR_PEG_TRAINING_SEEDS:-9351 4796 1788}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
output_root="${ATR_PEG_BC_OUTPUT:-results/manipulation_bc/external_peg_rl_demo_v1}"
.venv/bin/python scripts/pretrain_external_peg_bc.py \
  --data "${ATR_PEG_BC_DATA:?ATR_PEG_BC_DATA is required}" \
  --output "${output_root}/seed_${seed}/best.pt" \
  --seed "${seed}" \
  --count "${ATR_PEG_BC_COUNT:-1000}" \
  --epochs "${ATR_PEG_BC_EPOCHS:-100}" \
  --batch-size "${ATR_PEG_BC_BATCH_SIZE:-2048}" \
  --learning-rate "${ATR_PEG_BC_LR:-0.0003}"
