#!/usr/bin/env bash
#SBATCH --job-name=atr-peg-router-eval
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=results/slurm/peg_router_eval_%A_%a.out
#SBATCH --error=results/slurm/peg_router_eval_%A_%a.err

set -euo pipefail
mkdir -p results/slurm "${ATR_PEG_EVAL_OUTPUT:?set ATR_PEG_EVAL_OUTPUT}"
NOMINAL_ARGS=()
IFS=':' read -r -a nominal_checkpoints <<< "${ATR_PEG_NOMINAL_CHECKPOINTS:?set ATR_PEG_NOMINAL_CHECKPOINTS}"
for checkpoint in "${nominal_checkpoints[@]}"; do
  NOMINAL_ARGS+=(--nominal-checkpoint "${checkpoint}")
done
FORWARD_ARGS=()
if [[ -n "${ATR_PEG_FORWARD_CHECKPOINTS:-}" ]]; then
  IFS=':' read -r -a forward_checkpoints <<< "${ATR_PEG_FORWARD_CHECKPOINTS}"
  for checkpoint in "${forward_checkpoints[@]}"; do
    FORWARD_ARGS+=(--forward-checkpoint "${checkpoint}")
  done
fi
REVERSE_ARGS=()
if [[ -n "${ATR_PEG_REVERSE_CHECKPOINTS:-}" ]]; then
  IFS=':' read -r -a reverse_checkpoints <<< "${ATR_PEG_REVERSE_CHECKPOINTS}"
  for checkpoint in "${reverse_checkpoints[@]}"; do
    REVERSE_ARGS+=(--reverse-checkpoint "${checkpoint}")
  done
fi
ROUTER_ARGS=()
if [[ -n "${ATR_PEG_ROUTER_CHECKPOINT:-}" ]]; then
  ROUTER_ARGS+=(
    --router-checkpoint "${ATR_PEG_ROUTER_CHECKPOINT}"
    --router-metadata "${ATR_PEG_ROUTER_METADATA:?set ATR_PEG_ROUTER_METADATA}"
  )
fi
.venv/bin/python scripts/evaluate_external_peg_router.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --method "${ATR_PEG_EVAL_METHOD:?set ATR_PEG_EVAL_METHOD}" \
  --output-dir "${ATR_PEG_EVAL_OUTPUT}" \
  --seed-base "${ATR_PEG_EVAL_SEED_BASE:-425000000}" \
  --episodes "${ATR_PEG_EVAL_EPISODES:-64}" \
  --num-envs "${ATR_PEG_EVAL_NUM_ENVS:-64}" \
  --steps "${ATR_PEG_EVAL_STEPS:-160}" \
  "${NOMINAL_ARGS[@]}" \
  "${FORWARD_ARGS[@]}" \
  "${REVERSE_ARGS[@]}" \
  "${ROUTER_ARGS[@]}"
