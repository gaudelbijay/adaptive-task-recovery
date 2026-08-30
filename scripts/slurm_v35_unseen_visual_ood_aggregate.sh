#!/usr/bin/env bash
#SBATCH --job-name=atr-v35-unseen-aggregate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/v35_unseen_aggregate_%j.out
#SBATCH --error=results/slurm/v35_unseen_aggregate_%j.err

set -euo pipefail
: "${ATR_V35_UNSEEN_CONFIG:?set ATR_V35_UNSEEN_CONFIG}"
output="${ATR_V35_UNSEEN_AGGREGATE:-results/paper/v35_confirmatory_unseen_visual_ood_v1/aggregate.json}"
mkdir -p results/slurm
.venv/bin/python scripts/aggregate_selected_visual_causal_ood.py \
  --config "${ATR_V35_UNSEEN_CONFIG}" \
  --results-root "${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}" \
  --output "${output}"
