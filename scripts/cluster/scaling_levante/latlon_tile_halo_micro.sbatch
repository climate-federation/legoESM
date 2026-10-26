#!/bin/bash -l
#SBATCH --job-name=ll_halo_micro
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
#SBATCH --output=ll_halo_micro.%j.log
# Times the packed tile halo against the per-field one on the same devices.
# The count is already known (60 -> 10 for six fields); the open question is
# whether the TIME follows, which is what decides whether packing the
# remaining exchanges is worth building. See the script header.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
export NCCL_MIN_NCHANNELS=8 NCCL_MAX_NCHANNELS=8 NCCL_P2P_NET_CHUNKSIZE=131072
for FIELDS in 6 12; do
  echo "=== fields=$FIELDS ==="
  srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node=4 --gpu-bind=none \
    "$PY" scripts/bench/bench_latlon_tile_halo_micro.py \
      --p-lat 2 --p-lon 2 --n-lat 512 --n-lon 1024 --nlev 26 \
      --fields "$FIELDS" || exit 1
done
echo DONE
