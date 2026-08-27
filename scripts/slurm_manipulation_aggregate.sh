#!/usr/bin/env bash
#SBATCH --job-name=atr-manip-agg
#SBATCH --partition=compute-v2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=results/slurm/manip_agg_%j.out
#SBATCH --error=results/slurm/manip_agg_%j.err

set -euo pipefail

ATR_MANIP_CONFIG="${ATR_MANIP_CONFIG:-configs/manipulation_ppo_v1.json}"
ATR_MANIP_OUTPUT="${ATR_MANIP_OUTPUT:-results/manipulation_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/aggregate_manipulation_results.py \
  --config "${ATR_MANIP_CONFIG}" \
  --output "${ATR_MANIP_OUTPUT}"

"${ATR_PYTHON}" scripts/plot_manipulation_results.py \
  --config "${ATR_MANIP_CONFIG}" \
  --output "${ATR_MANIP_OUTPUT}"
