#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cpuladder
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/compile_ladder_cpu_%j.log
# The CPU arm of the compile ladder: does the PROGRAM SHAPE alone explain the
# 24-GPU stall? Fake devices, no GPU work. This can confirm that the frontend
# and partitioner cost explodes with the window count; it can NEVER rule out
# GPU code generation, autotuning or NCCL layout.
#
# Matched to the job that stalled (9694183): C24, km=5, one acoustic sub-step.
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1
export LEGOESM_JIT_CACHE_DIR=""
cd "$REPO"
echo "host $(hostname)  sha $(git rev-parse HEAD)  dirty $(git status --porcelain | wc -l)"
timeout 20000 $PY scripts/validate/fv3_native/compile_rank_ladder.py \
  --backend cpu --n 24 --km 5 --pad 5 --n-split 1 2>&1
echo "rc=$?"
echo CPULADDER_DONE
