#!/usr/bin/env bash
#SBATCH --job-name=atr-checkpoint-audit
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm/checkpoint_audit_%j.out
#SBATCH --error=results/slurm/checkpoint_audit_%j.err

set -euo pipefail
mkdir -p results/slurm
: "${ATR_AUDIT_CONFIG:?set ATR_AUDIT_CONFIG}"
: "${ATR_AUDIT_OUTPUT_ROOT:?set ATR_AUDIT_OUTPUT_ROOT}"
audit_args=(
  --config "${ATR_AUDIT_CONFIG}"
  --output "${ATR_AUDIT_OUTPUT_ROOT}"
)
if [[ -n "${ATR_AUDIT_FILE:-}" ]]; then
  audit_args+=(--audit-output "${ATR_AUDIT_FILE}")
elif [[ "${ATR_AUDIT_CONFIG}" == *confirm_append.json ]]; then
  ATR_AUDIT_EXPERIMENT=$(.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["name"])' \
    "${ATR_AUDIT_CONFIG}")
  audit_args+=(
    --audit-output \
    "${ATR_AUDIT_OUTPUT_ROOT}/${ATR_AUDIT_EXPERIMENT}/checkpoint_audit_confirmatory.json"
  )
fi
.venv/bin/python scripts/audit_training_checkpoints.py "${audit_args[@]}"
