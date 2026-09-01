#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-convert
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/peg_convert_%j.out
#SBATCH --error=results/slurm/peg_convert_%j.err

set -euo pipefail
mkdir -p results/slurm
.venv/bin/python scripts/convert_external_peg_demonstrations.py \
  --source "${ATR_PEG_DEMO_SOURCE:-results/external_demos/PegInsertionSide-v1/rl/trajectory.h5}" \
  --output "${ATR_PEG_DEMO_OUTPUT:-results/external_demos/PegInsertionSide-v1/rl/state_actions_v1.npz}" \
  --count "${ATR_PEG_DEMO_COUNT:-1000}" \
  --num-envs "${ATR_PEG_DEMO_CONVERSION_ENVS:-1024}"
