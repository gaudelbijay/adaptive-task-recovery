#!/usr/bin/env bash
#SBATCH --job-name=atr-factor-dispatch
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=results/slurm/factor_dispatch_%A_%a.out
#SBATCH --error=results/slurm/factor_dispatch_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/router/v18_factorized_dispatch
if (( SLURM_ARRAY_TASK_ID < 3 )); then
  FAMILY=causal_gru
  SEED=${SLURM_ARRAY_TASK_ID}
else
  FAMILY=static_mlp
  SEED=$((SLURM_ARRAY_TASK_ID - 3))
fi
.venv/bin/python scripts/calibrate_factorized_sweep_dispatch.py \
  --checkpoint "results/router/v17_full_geometry_composition/${FAMILY}_seed${SEED}.pt" \
  --data results/router/v6_instant96_dagger_full.npz \
  --metadata results/router/v6_instant96_dagger_full.json \
  --output "results/router/v18_factorized_dispatch/${FAMILY}_seed${SEED}.pt" \
  > "results/router/v18_factorized_dispatch/${FAMILY}_seed${SEED}.json"
