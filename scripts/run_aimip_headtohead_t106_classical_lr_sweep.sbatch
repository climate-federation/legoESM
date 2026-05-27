#!/bin/bash
#SBATCH --job-name=h2h_cls_lr
#SBATCH --output=results/aimip_t106_headtohead_lr%a/slurm_logs/classical_lr%a_%A_%a.out
#SBATCH --error=results/aimip_t106_headtohead_lr%a/slurm_logs/classical_lr%a_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 classical head-to-head LR mini-sweep.  Insurance copies of the
# live job 8033508, with 4 different learning rates so we can pick the
# best-performing classical run.  Same base config, same physics
# (zhang_mcfarlane / edmf / hines / sundqvist / xu_randall), same
# 4-epoch / multi-step / RRTMGP setup; only aimip_lr differs.
#
# Mapping:
#   task 1 -> aimip_lr=1.5e-4   (half base)
#   task 2 -> aimip_lr=3.0e-4   (base, matches 8033508)
#   task 3 -> aimip_lr=6.0e-4   (2x base)
#   task 4 -> aimip_lr=1.2e-3   (4x base)
#
# Each task writes to its own results/aimip_t106_headtohead_lr{N}/
# directory so scorecards do not collide.
#
# Submit with:
#   sbatch --array=1-4 scripts/run_aimip_headtohead_t106_classical_lr_sweep.sbatch

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUITE="config/aimip/t106_headtohead/suite_lr${TASK_ID}.yaml"
OUT_DIR="results/aimip_t106_headtohead_lr${TASK_ID}"

mkdir -p "${OUT_DIR}/slurm_logs"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME="${HOME:-/burg-archive/home/pg2328}"

# Share XLA compile cache across the 4 array tasks: the classical
# graph is identical apart from the LR scalar baked into the optax
# schedule, so the first task pays the ~6 h compile cost and the rest
# should hit the cache.
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_classical_lr"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-lr-sweep] $(date)  task=${TASK_ID}  suite=${SUITE}  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run_aimip.py \
    --suite "${SUITE}" \
    --variants classical

echo "[h2h-lr-sweep] task=${TASK_ID} done $(date)"
ls -la "${OUT_DIR}/aimip_scorecard.json" 2>&1 || true
