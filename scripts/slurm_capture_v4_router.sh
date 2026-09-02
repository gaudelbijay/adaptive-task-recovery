#!/usr/bin/env bash
#SBATCH --job-name=atr-v4-capture
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/v4_capture_%j.out
#SBATCH --error=results/slurm/v4_capture_%j.err

# Reproduces the three episodes in media/demos/v4-router-montage.gif.
#
# Settings match the matched-input evaluation exactly, including the disabled
# factorized sweep dispatch, so the captured episodes are the evaluated ones
# with rendering added. Each capture records the step at which its episode
# resolves; the montage builder holds that frame, because scoring stops at
# first resolution and later motion is not measured.
#
# Task index = seed_index * 5 + condition_index, with conditions ordered
# nominal, ejection, permanent_block, temporary_block, reverse_ejection.

set -euo pipefail
mkdir -p results/slurm results/v4_capture2 results/v4_capture2t
export PYTHONUNBUFFERED=1

REVERSE="results/manipulation_ppo/learned_recovery_v4_reverse_handoff_continuation_v2/reverse_ejection_state_handoff_continuation_v2/seed_4796/best.pt"
COMMON=(
  --router-checkpoint results/router/v18_factorized_dispatch/causal_gru_seed0.pt
  --router-metadata results/router/v6_instant96_dagger_full.json
  --reverse-state-checkpoint "${REVERSE}"
  --seed-base 347000000 --episodes 8 --num-envs 8 --confirmation-steps 1
  --safe-hold-until-step 40 --safe-hold-start-step 1
  --defer-action-mode retreat_to_reset --router-query-every-step
  --terminate-score-on-first-resolution
  --nominal-ensemble --nominal-ensemble-reduction mean --temporary-policy-index 2
)

# Reverse ejection (4) and permanent blockage (2) from environment index 0.
for TASK in 4 2; do
  PYTHONPATH=scripts .venv/bin/python scripts/evaluate_v4_learned_option_router.py \
    --task-index "${TASK}" "${COMMON[@]}" \
    --output-dir results/v4_capture2 \
    --capture-video results/v4_capture2/router --capture-env-index 0
done

# Temporary blockage (3) from environment index 2: index 0 was one of the
# 15.6% of episodes that fail, and the montage builder refuses a failed panel.
PYTHONPATH=scripts .venv/bin/python scripts/evaluate_v4_learned_option_router.py \
  --task-index 3 "${COMMON[@]}" \
  --output-dir results/v4_capture2t \
  --capture-video results/v4_capture2t/router --capture-env-index 2
