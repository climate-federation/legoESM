#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=mpas_c6_smoke
#SBATCH --partition=short
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/mpas_cam6_ref_terrainC/smoke_1d/slurm_%j.out
# MPAS res6 L32 CAM6 production deck (config/amip/amip_production.yaml) on the
# decision-C terrain product, 1 simulated day, ONE process on ONE A100:
# Ginsburg has no mpi4jax, so the Voronoi lane cannot run multi-rank here.
# Measures the per-day rate and verifies the lane runs on the product
# (resolved config written to the run dir) before the 60-day reference.
set -uo pipefail
WT=${WT:-/burg-archive/glab/users/pg2328/legoESM_codex_m8}
export LEGOESM_REPO="$WT"
export PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH="$WT/packages/core:$WT/packages/atmosphere:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ocean:$WT/packages/ml:$WT/packages/tools:$WT/src"
export AMIP_DATA=/burg-archive/glab/users/ac5006/legoESM/data/amip
export AMIP_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/legoesm_surfdata_c260716.nc
export CLM_SURFDATA=/burg-archive/glab/users/pg2328/legoESM/data/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc
export JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 LEGOESM_JIT_CACHE_DIR=""
# 40 GB A100: the compiled step needs >= 27.8 GiB live (XLA remat floor), so no
# 75 % preallocation; the Levante chain runs the same deck with these two.
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 TF_GPU_ALLOCATOR=cuda_malloc_async
OUT=${OUT:-/burg-archive/glab/users/pg2328/fv3_duo_gaps/mpas_cam6_ref_terrainC/smoke_1d}
cd "$WT"; echo "sha $(git rev-parse --short HEAD) dirty $(git status --porcelain | grep -v '^??' | wc -l) host $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
source "$WT/config/amip/amip_production.ginsburg.sh"
"$PY" -u scripts/run/run_amip.py --config config/amip/amip_production.yaml \
  "${AMIP_PATH_FLAGS[@]}" --days "${DAYS:-1}" --checkpoint-days 1 --output "$OUT"
echo "rc=$?"
