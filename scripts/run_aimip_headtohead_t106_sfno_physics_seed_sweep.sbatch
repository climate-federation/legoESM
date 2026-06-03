#!/bin/bash
#SBATCH --job-name=h2h_sfn_s
#SBATCH --output=results/aimip_t106_headtohead_sfno_physics_s%a/slurm_logs/sfno_physics_s%a_%A_%a.out
#SBATCH --error=results/aimip_t106_headtohead_sfno_physics_s%a/slurm_logs/sfno_physics_s%a_%A_%a.err
#SBATCH --account=glab
#SBATCH --partition=burst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=72:00:00

# T106 sfno_physics head-to-head seed mini-sweep.  Insurance copies of
# the live job 8033427 (sfno_seed=0 by default, 14 h walltime) with a
# fresh 72 h walltime.  Same base config, same SFNO architecture
# (embed_dim=32, n_blocks=2, ~23.7 M params), same 4-epoch /
# multi-step / RRTMGP setup; only the SFNO init seed (sfno_seed)
# differs.
#
# Mapping:
#   task 0 -> sfno_seed=0   (true insurance copy of live job 8033427)
#   task 1 -> sfno_seed=1
#   task 2 -> sfno_seed=2
#   task 3 -> sfno_seed=3
#
# Submit with:
#   sbatch --array=0-3 scripts/run_aimip_headtohead_t106_sfno_physics_seed_sweep.sbatch

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "$REPO_ROOT"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SUITE="config/aimip/t106_headtohead/suite_sfno_physics_s${TASK_ID}.yaml"
OUT_DIR="results/aimip_t106_headtohead_sfno_physics_s${TASK_ID}"

mkdir -p "${OUT_DIR}/slurm_logs"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export HOME="${HOME:-/burg-archive/home/pg2328}"

# Shared XLA compile cache across the 4 seeds.  The SFNO init weights
# are the only thing that varies; HLO graph shape is identical so the
# first task warms the cache and the other three should reuse it.
export JAX_COMPILATION_CACHE_DIR="${REPO_ROOT}/.cache/jax_compile_cache_h2h_sfno_physics_seed"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

echo "[h2h-sfno-sweep] $(date)  task=${TASK_ID}  suite=${SUITE}  host=$(hostname)"
nvidia-smi -L 2>&1 | head -5 || echo "  (nvidia-smi unavailable)"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
    -u scripts/run_aimip.py \
    --suite "${SUITE}" \
    --variants sfno_physics

echo "[h2h-sfno-sweep] task=${TASK_ID} done $(date)"
ls -la "${OUT_DIR}/aimip_scorecard.json" 2>&1 || true
