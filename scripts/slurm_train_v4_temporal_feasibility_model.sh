#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-fit
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/v4_fit_%j.out
#SBATCH --error=results/slurm/v4_fit_%j.err

set -euo pipefail
mkdir -p results/slurm results/models
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/train_v4_temporal_feasibility_model.py \
  --late-horizon "${ATR_LATE_HORIZON:-32}" \
  --model-output "${ATR_MODEL_OUTPUT:-results/models/v4_temporal_dinov2_two_stage_v7.joblib}" \
  --metadata-output "${ATR_METADATA_OUTPUT:-results/models/v4_temporal_dinov2_two_stage_v7.json}"
