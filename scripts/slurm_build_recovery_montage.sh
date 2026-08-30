#!/usr/bin/env bash
#SBATCH --job-name=atr-recovery-montage
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/recovery_montage_%j.out
#SBATCH --error=results/slurm/recovery_montage_%j.err

set -euo pipefail
: "${ATR_MONTAGE_METHOD:?set ATR_MONTAGE_METHOD}"
: "${ATR_MONTAGE_SEED:?set ATR_MONTAGE_SEED}"
: "${ATR_MONTAGE_OUTPUT:?set ATR_MONTAGE_OUTPUT}"
.venv/bin/python scripts/build_recovery_montage.py \
  --videos "${ATR_MONTAGE_VIDEOS:-results/visual_recovery_ppo/videos}" \
  --method "${ATR_MONTAGE_METHOD}" --seed "${ATR_MONTAGE_SEED}" \
  --output "${ATR_MONTAGE_OUTPUT}" --strict-removal-labels
