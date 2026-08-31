#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-router-fit
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/v4_router_fit_%A_%a.out
#SBATCH --error=results/slurm/v4_router_fit_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/router/v4_causal_router_v1
export PYTHONUNBUFFERED=1
read -r -a ROUTER_MODELS <<< "${ATR_ROUTER_MODELS:-causal_gru static_mlp unstructured_gru}"
.venv/bin/python scripts/train_v4_causal_option_router.py \
  --data "${ATR_ROUTER_DATA:-results/router/v4_option_prefixes_train_v1.npz}" \
  --metadata "${ATR_ROUTER_METADATA:-results/router/v4_option_prefixes_train_v1.json}" \
  --output-dir "${ATR_ROUTER_OUTPUT:-results/router/v4_causal_router_v1}" \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --models "${ROUTER_MODELS[@]}"
