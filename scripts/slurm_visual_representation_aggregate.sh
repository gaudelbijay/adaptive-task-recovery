#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-probe-agg
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_probe_aggregate_%j.out
#SBATCH --error=results/slurm/visual_probe_aggregate_%j.err

set -euo pipefail
ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_REPRESENTATION_AGGREGATE_FILENAME="${ATR_REPRESENTATION_AGGREGATE_FILENAME:-representation_probe_aggregate.json}"
ATR_REPRESENTATION_PROBE_FILENAME="${ATR_REPRESENTATION_PROBE_FILENAME:-representation_probe.json}"
"${ATR_PYTHON}" scripts/aggregate_visual_representation_probes.py \
  --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT}" \
  --filename "${ATR_REPRESENTATION_AGGREGATE_FILENAME}" \
  --probe-filename "${ATR_REPRESENTATION_PROBE_FILENAME}"
