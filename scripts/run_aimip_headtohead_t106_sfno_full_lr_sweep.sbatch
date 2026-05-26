#!/bin/bash
#SBATCH --job-name=h2h_sfn_full
#SBATCH --output=results/aimip_t106_headtohead_sfno_full_lr%a/slurm_logs/sfno_full_lr%a_%A_%a.out
#SBATCH --error=results/aimip_t106_headtohead_sfno_full_lr%a/slurm_logs/sfno_full_lr%a_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 sfno_full head-to-head LR mini-sweep.  Trains SFNO as a full
# atmospheric emulator (no dycore, no physics tendency) via the new
# train_sfno_full_spectral path.  Same SFNO arch as the head-to-head
# sfno_physics variant (embed_dim=32, n_blocks=2) so the trained-
# model size is directly comparable; the macro time step is
# dt_sfno=21600s (6 h, standard FourCastNet/SFNO cadence) -> 4 SFNO
# steps per 24 h segment in the multi-step (24, 48, 72) h schedule.
#
# Mapping:
#   task 1 -> aimip_lr=5.0e-4   (1.67x base; modest push)
#   task 2 -> aimip_lr=1.0e-3   (3.3x base; aggressive, tests
#                                whether Muon can drive faster
#                                convergence on the end-to-end
#                                emulator)
#
# Both use sfno_seed=0 so the only varying knob is LR.
#
# Submit with:
#   sbatch --array=1-2 scripts/run_aimip_headtohead_t106_sfno_full_lr_sweep.sbatch

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
SUITE="config/aimip/t106_headtohead/suite_sfno_full_lr${TASK_ID}.yaml"
OUT_DIR="results/aimip_t106_headtohead_sfno_full_lr${TASK_ID}"

mkdir -p "${OUT_DIR}/slurm_logs"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME="${HOME:-/burg-archive/home/pg2328}"

# Dedicated XLA compile cache for the full-emulator graph (different
# HLO shape than sfno_physics, which couples through the dycore, so
# do not share with the sfno_seed cache).
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_sfno_full_lr"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-sfno_full-lr-sweep] $(date)  task=${TASK_ID}  suite=${SUITE}  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run_aimip.py \
    --suite "${SUITE}" \
    --variants sfno_full

echo "[h2h-sfno_full-lr-sweep] task=${TASK_ID} done $(date)"
ls -la "${OUT_DIR}/aimip_scorecard.json" 2>&1 || true
