#!/usr/bin/env bash
#SBATCH --job-name=atr-v3-confirm-gate
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/visual_confirm_gate_%j.out
#SBATCH --error=results/slurm/visual_confirm_gate_%j.err

set -euo pipefail
: "${ATR_CONFIRM_VISUAL_AGGREGATE:?set ATR_CONFIRM_VISUAL_AGGREGATE}"
: "${ATR_CONFIRM_STATE_AGGREGATE:?set ATR_CONFIRM_STATE_AGGREGATE}"
: "${ATR_CONFIRM_RELEASE_JOB_IDS:?set ATR_CONFIRM_RELEASE_JOB_IDS}"
ATR_PYTHON="${ATR_PYTHON:-.venv/bin/python}"

"${ATR_PYTHON}" scripts/check_visual_confirmatory_gate.py \
  --visual-aggregate "${ATR_CONFIRM_VISUAL_AGGREGATE}" \
  --state-aggregate "${ATR_CONFIRM_STATE_AGGREGATE}" \
  --visual-method event_reward_learned_progress_adaptive_visual_ppo \
  --state-method event_reward_safe_adaptive_state_ppo \
  --new-seeds 71064 84293

for ATR_CONFIRM_JOB_ID in ${ATR_CONFIRM_RELEASE_JOB_IDS//:/ }; do
  scontrol release "${ATR_CONFIRM_JOB_ID}"
  echo "released ${ATR_CONFIRM_JOB_ID} after frozen V5 screening gate"
done
