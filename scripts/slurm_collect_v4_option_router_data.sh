#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-router-data
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/v4_router_data_%j.out
#SBATCH --error=results/slurm/v4_router_data_%j.err

set -euo pipefail
mkdir -p results/slurm results/router
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/collect_v4_option_router_data.py \
  --output "${ATR_ROUTER_DATA:-results/router/v4_option_prefixes_train_v1.npz}" \
  --metadata-output "${ATR_ROUTER_METADATA:-results/router/v4_option_prefixes_train_v1.json}" \
  --batches-per-kind "${ATR_ROUTER_BATCHES_PER_KIND:-20}" \
  --seed-base "${ATR_ROUTER_SEED_BASE:-310000000}" \
  --conditions "${ATR_ROUTER_CONDITIONS:-nominal,ejection,permanent_block,temporary_block,reverse_ejection}" \
  --behavior "${ATR_ROUTER_BEHAVIOR:-zero}" \
  --policy-index "${ATR_ROUTER_POLICY_INDEX:-0}" \
  --onset-min "${ATR_ROUTER_ONSET_MIN:-0}" \
  --onset-max "${ATR_ROUTER_ONSET_MAX:-8}" \
  --force-scale-min "${ATR_ROUTER_FORCE_MIN:-0.85}" \
  --force-scale-max "${ATR_ROUTER_FORCE_MAX:-1.15}" \
  --return-delay-min "${ATR_ROUTER_RETURN_MIN:-24}" \
  --return-delay-max "${ATR_ROUTER_RETURN_MAX:-36}" \
  --horizon "${ATR_ROUTER_HORIZON:-96}"
