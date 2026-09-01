#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-ppo-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=results/slurm/peg_ppo_eval_%A_%a.out
#SBATCH --error=results/slurm/peg_ppo_eval_%A_%a.err

set -euo pipefail
mkdir -p results/slurm results/a_plus_audit/external_peg_nominal_ppo_v1
read -r -a seeds <<< "${ATR_PEG_TRAINING_SEEDS:-9351 4796 1788}"
seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
run_root="${ATR_PEG_RUN_ROOT:-results/manipulation_ppo/external_peg_nominal_ppo_v1/official_state_ppo_nominal}"
audit_dir="${ATR_PEG_AUDIT_DIR:-results/a_plus_audit/external_peg_nominal_ppo_v1}"
mkdir -p "${audit_dir}"
run_dir="${run_root}/seed_${seed}"
.venv/bin/python scripts/evaluate_external_peg_ppo.py \
  --checkpoint "${run_dir}/${ATR_PEG_CHECKPOINT_NAME:-best.pt}" \
  --episodes "${ATR_PEG_EVAL_EPISODES:-192}" \
  --num-envs "${ATR_PEG_EVAL_NUM_ENVS:-64}" \
  --seed-base "${ATR_PEG_EVAL_SEED_BASE:-421000000}" \
  --steps "${ATR_PEG_EVAL_STEPS:-100}" \
  --output "${audit_dir}/seed_${seed}.json"
