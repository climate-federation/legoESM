#!/bin/bash
#SBATCH --job-name=marbl_col_exps
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out
set -euo pipefail
cd /work/bb1596/b384036/diffESM/legoesm_ggn_newcopy
eval "$(micromamba shell hook --shell bash)"
micromamba activate /work/bb1596/b384036/b384036/envs/climateval_gpu
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
E=scripts/marbl_column_experiments.py
python $E spinup --years 2
python $E grad-check --days 30
python $E sensitivity --days 365
python $E twin --days 365 --k 6 --iters 60 --seed 0
