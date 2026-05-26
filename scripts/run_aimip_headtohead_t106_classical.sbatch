#!/bin/bash
#SBATCH --job-name=h2h_classical
#SBATCH --output=results/aimip_t106_headtohead/slurm_logs/classical_%j.out
#SBATCH --error=results/aimip_t106_headtohead/slurm_logs/classical_%j.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 head-to-head: trains the classical variant with the new
# multi-day + bias + spectral-CRPS loss and RRTMGP radiation.
# Configs: config/aimip/t106_headtohead/(base.yaml + suite.yaml + variant_classical.yaml)

set -euo pipefail
REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

mkdir -p results/aimip_t106_headtohead/slurm_logs

export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:${PYTHONPATH:-}
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    scripts/run_aimip.py \
    --suite config/aimip/t106_headtohead/suite.yaml \
    --variants classical
