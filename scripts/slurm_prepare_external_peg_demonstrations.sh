#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-demos
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm/peg_demos_%j.out
#SBATCH --error=results/slurm/peg_demos_%j.err

set -euo pipefail
mkdir -p results/slurm
export GIT_PYTHON_REFRESH=quiet
source_path="${ATR_PEG_DEMO_SOURCE:-results/external_demos/PegInsertionSide-v1/rl/trajectory.h5}"
.venv/bin/python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${source_path}" \
  --obs-mode state \
  --sim-backend physx_cpu \
  --target-control-mode pd_joint_delta_pos \
  --use-first-env-state \
  --save-traj \
  --record-rewards \
  --count "${ATR_PEG_DEMO_COUNT:-1000}" \
  --num-envs "${ATR_PEG_DEMO_WORKERS:-16}"
