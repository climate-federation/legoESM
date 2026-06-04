#!/usr/bin/env bash
#SBATCH --job-name=rrtmgp_shard_bench
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=results/rrtmgp_shard_bench/slurm-%j.out
#SBATCH --error=results/rrtmgp_shard_bench/slurm-%j.err
#
# Validate the two rrtmgp-mpas-perf levers (commit afcdb9e2) on a 2-GPU node:
#   1. compute_fp32  -- fp32 optics/RTE vs fp64 baseline: speedup + max-rel-dT
#                       (correctness gate rel < 1e-2 in the script).
#   2. column_mesh   -- shard nCells across both RTX 8000s: speedup + bit-match
#                       vs single-device (gate rel < 1e-6 -> MATCH/MISMATCH).
# Isolates the radiation kernel (no driver/dynamics). res=4 -> 2562 cells,
# divisible by 2 devices, so the sharded branch runs (not SKIPPED).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd "$REPO"
mkdir -p results/rrtmgp_shard_bench

PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1

echo "host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "--- bench ---"
"$PY" scripts/tmp/_diag_rrtmgp_shard_bench.py --res 4 --nlev 30
echo "--- done rc=$? ---"
