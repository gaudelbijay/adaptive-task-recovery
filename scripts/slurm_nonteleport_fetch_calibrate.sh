#!/bin/bash
#SBATCH --job-name=atr-fetch-cal
#SBATCH --partition=compute-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-9
#SBATCH --output=results/slurm/nonteleport_fetch_cal_%A_%a.out
#SBATCH --error=results/slurm/nonteleport_fetch_cal_%A_%a.err

set -euo pipefail
cd /home/bgaudel/adaptive-task-recovery
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
if (( SLURM_ARRAY_TASK_ID < 5 )); then
  intervention=none
  seed="${SLURM_ARRAY_TASK_ID}"
else
  intervention=cracker_box_destroyed
  seed="$((SLURM_ARRAY_TASK_ID - 5))"
fi
.venv/bin/python scripts/calibrate_nonteleport_fetch_vision.py \
  --seed "${seed}" --intervention "${intervention}" \
  --output "results/nonteleport_fetch/calibration/${intervention}_seed_${seed}.json"
