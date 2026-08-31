#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-history-audit
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/v3_history_audit_%j.out
#SBATCH --error=results/slurm/v3_history_audit_%j.err

set -euo pipefail
mkdir -p results/slurm results/router/v17_full_geometry_composition
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/audit_temporal_composition_ablations.py \
  --data "${ATR_ROUTER_DATA:-results/router/v6_instant96_dagger_full.npz}" \
  --checkpoint "${ATR_ROUTER_CHECKPOINT:-results/router/v17_full_geometry_composition/causal_gru_seed0.pt}" \
  --output "${ATR_AUDIT_OUTPUT:-results/router/v17_full_geometry_composition/history_ablation.json}"
