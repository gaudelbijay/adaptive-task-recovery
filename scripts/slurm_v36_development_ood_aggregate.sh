#!/usr/bin/env bash
#SBATCH --job-name=atr-v36-development-aggregate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v36_development_aggregate_%j.out
#SBATCH --error=results/slurm/v36_development_aggregate_%j.err

set -euo pipefail
: "${ATR_V36_DEVELOPMENT_CONFIG:?set ATR_V36_DEVELOPMENT_CONFIG}"
output="${ATR_V36_DEVELOPMENT_AGGREGATE:-results/paper/v36_smoke_development_ood_v1/aggregate.json}"
mkdir -p results/slurm
.venv/bin/python scripts/aggregate_selected_visual_causal_ood.py \
  --config "${ATR_V36_DEVELOPMENT_CONFIG}" \
  --results-root "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --output "${output}"
