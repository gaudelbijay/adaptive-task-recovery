#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-learning-plot
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/visual_learning_plot_%j.out
#SBATCH --error=results/slurm/visual_learning_plot_%j.err

set -euo pipefail
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"
"${ATR_PYTHON}" scripts/plot_visual_recovery_learning.py \
  --config configs/visual_recovery_ppo_gate_v2_event_reward.json \
  --config configs/visual_recovery_dagger_ablation_v7_event_reward.json \
  --config configs/visual_recovery_progress_dagger_v6_event_reward.json \
  --output-root results/visual_recovery_ppo \
  --figure-stem "${ATR_LEARNING_FIGURE_STEM:-media/results/v3_visual_recovery_learning}"
