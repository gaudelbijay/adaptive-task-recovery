#!/usr/bin/env bash
#SBATCH --job-name=atr-visual-task-probe-agg
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_task_probe_aggregate_%j.out
#SBATCH --error=results/slurm/visual_task_probe_aggregate_%j.err

set -euo pipefail
ATR_VISUAL_CONFIG="${ATR_VISUAL_CONFIG:?set ATR_VISUAL_CONFIG}"
ATR_VISUAL_OUTPUT="${ATR_VISUAL_OUTPUT:-results/visual_recovery_ppo}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
ATR_TASK_REPRESENTATION_AGGREGATE_FILENAME="${ATR_TASK_REPRESENTATION_AGGREGATE_FILENAME:-task_representation_probe_aggregate.json}"
"${ATR_PYTHON}" scripts/aggregate_visual_task_representation_probes.py \
  --config "${ATR_VISUAL_CONFIG}" --output "${ATR_VISUAL_OUTPUT}" \
  --filename "${ATR_TASK_REPRESENTATION_AGGREGATE_FILENAME}"
