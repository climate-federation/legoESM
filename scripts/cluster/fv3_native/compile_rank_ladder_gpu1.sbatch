#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=gpuladder1
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/compile_ladder_gpu1_%j.log
# The single most informative number for the 12 h stall: how long does ONE GPU
# take to compile the same window step the CPU ladder compiles in ~12 s?
#
# The CPU ladder already showed compile time FLAT from 1 to 24 devices
# (11.6 -> 13.1 s) with the program barely growing, so program shape is
# exonerated. If one GPU is also seconds-to-minutes, the compiler is exonerated
# too and the stall is in the collectives. If one GPU is already hours, it is
# GPU code generation or autotuning and no distributed forensics are needed.
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LEGOESM_JIT_CACHE_DIR=""
export JAX_LOG_COMPILES=1
cd "$REPO"
echo "host $(hostname)  sha $(git rev-parse HEAD)  dirty $(git status --porcelain | wc -l)"
nvidia-smi --query-gpu=name --format=csv,noheader
timeout 6600 $PY scripts/validate/fv3_native/compile_rank_ladder.py \
  --backend gpu --devices 1 --n 24 --km 5 --pad 5 --n-split 1 2>&1
echo "rc=$?"
echo GPULADDER1_DONE
