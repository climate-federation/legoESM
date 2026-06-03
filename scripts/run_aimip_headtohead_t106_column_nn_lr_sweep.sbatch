#!/bin/bash
#SBATCH --job-name=h2h_cnn_lr
#SBATCH --output=results/aimip_t106_headtohead_column_nn_lr%a/slurm_logs/column_nn_lr%a_%A_%a.out
#SBATCH --error=results/aimip_t106_headtohead_column_nn_lr%a/slurm_logs/column_nn_lr%a_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 column_nn head-to-head LR mini-sweep.  Targets the diagnosis
# from the live job 8033426: epoch 0 produced grad_norm=1.88e+02 with
# the default LR=3.0e-4 and grad_clip=1.0, so every step was hard-
# clipped to unit norm.  These two replicates lower the LR so the
# clipper saturates less often and the effective step direction is
# the true gradient.
#
# Mapping:
#   task 1 -> aimip_lr=1.0e-4   (3x lower than base)
#   task 2 -> aimip_lr=5.0e-5   (6x lower than base; conservative floor)
#
# Both use nn_seed=0 (matches the live job and seed-sweep s0 baseline)
# so the only varying knob is LR.
#
# Submit with:
#   sbatch --array=1-2 scripts/run_aimip_headtohead_t106_column_nn_lr_sweep.sbatch

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUITE="config/aimip/t106_headtohead/suite_column_nn_lr${TASK_ID}.yaml"
OUT_DIR="results/aimip_t106_headtohead_column_nn_lr${TASK_ID}"

mkdir -p "${OUT_DIR}/slurm_logs"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME="${HOME:-/burg-archive/home/pg2328}"

# Share XLA compile cache with the column_nn seed sweep: HLO graph
# shape is identical (only LR scalar in the optax cosine schedule
# differs), so any cache hit reduces compile cost.
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_column_nn_seed"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-column_nn-lr-sweep] $(date)  task=${TASK_ID}  suite=${SUITE}  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run_aimip.py \
    --suite "${SUITE}" \
    --variants column_nn

echo "[h2h-column_nn-lr-sweep] task=${TASK_ID} done $(date)"
ls -la "${OUT_DIR}/aimip_scorecard.json" 2>&1 || true
