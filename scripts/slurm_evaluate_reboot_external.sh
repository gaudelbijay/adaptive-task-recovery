#!/usr/bin/env bash
#SBATCH --job-name=atr-reboot-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/reboot_eval_%A_%a.out
#SBATCH --error=results/slurm/reboot_eval_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/reboot
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/evaluate_reboot_causal_prefix.py \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --train-fraction "${ATR_REBOOT_TRAIN_FRACTION:-1.0}" \
  --output "results/reboot/${ATR_REBOOT_OUTPUT_PREFIX:-reboot_causal_prefix}_seed${SLURM_ARRAY_TASK_ID}.json"
