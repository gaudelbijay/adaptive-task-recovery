#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-nominal
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-31
#SBATCH --output=results/slurm/peg_nominal_%j.out
#SBATCH --error=results/slurm/peg_nominal_%j.err

set -euo pipefail
mkdir -p results/slurm results/a_plus_audit
export PYTHONUNBUFFERED=1
.venv/bin/python -X faulthandler scripts/audit_external_peg_nominal_controller.py \
  --episodes 1 \
  --seed-base "${ATR_PEG_NOMINAL_SEED_BASE:-421000100}" \
  --seed-offset "${SLURM_ARRAY_TASK_ID}" \
  --minimum-success-rate "${ATR_PEG_NOMINAL_MIN_SUCCESS:-0.75}" \
  > "results/a_plus_audit/external_peg_nominal_controller_v1_${SLURM_ARRAY_TASK_ID}.json"
