#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-router-data
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/peg_router_data_%A_%a.out
#SBATCH --error=results/slurm/peg_router_data_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/router/external_peg_prefixes_v1
read -r -a seeds <<< "${ATR_PEG_TRAINING_SEEDS:-9351 4796 1788}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
run_root="${ATR_PEG_RUN_ROOT:-results/manipulation_ppo/external_peg_nominal_ppo_v1/official_state_ppo_nominal}"
output_root="${ATR_PEG_ROUTER_DATA_ROOT:-results/router/external_peg_prefixes_v1}"
seed_base="${ATR_PEG_ROUTER_SEED_BASE:-421100000}"
mkdir -p "${output_root}"
checkpoint="${run_root}/seed_${seed}/best.pt"
.venv/bin/python scripts/collect_external_peg_router_data.py \
  --checkpoint "${checkpoint}" \
  --batches-per-kind "${ATR_PEG_ROUTER_BATCHES_PER_KIND:-4}" \
  --num-envs "${ATR_PEG_ROUTER_NUM_ENVS:-64}" \
  --horizon "${ATR_PEG_ROUTER_HORIZON:-96}" \
  --seed-base "$((seed_base + SLURM_ARRAY_TASK_ID * 100000))" \
  --output "${output_root}/seed_${seed}.npz" \
  --metadata-output "${output_root}/seed_${seed}.json"
