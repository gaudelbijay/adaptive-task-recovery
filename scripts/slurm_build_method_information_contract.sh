#!/usr/bin/env bash
#SBATCH --job-name=atr-method-contract
#SBATCH --partition=compute-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=results/slurm/method_contract_%j.out
#SBATCH --error=results/slurm/method_contract_%j.err

set -euo pipefail
mkdir -p results/slurm results/paper
.venv/bin/python scripts/build_method_information_contract.py \
  --config configs/paper_method_information_contract_v1.json \
  --output-prefix results/paper/method_information_contract_v1
