#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-summary
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/peg_summary_%j.out
#SBATCH --error=results/slurm/peg_summary_%j.err

set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/summarize_external_peg_ppo_competence.py \
  --config "${ATR_PEG_CONFIG:?ATR_PEG_CONFIG is required}" \
  --input-dir "${ATR_PEG_AUDIT_DIR:?ATR_PEG_AUDIT_DIR is required}" \
  --output "${ATR_PEG_SUMMARY:?ATR_PEG_SUMMARY is required}" \
  --fail-on-reject
