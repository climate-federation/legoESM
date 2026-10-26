#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=par32
#SBATCH --output=mpas_parity_levels.%j.log
#SBATCH --exclude=l50003,l50139,l50027,l50009,l50100
# The campaign's headline recommendation is to run the icosahedral core at 32
# vertical levels instead of 26. The ladder that measured that speed-up is a
# timing harness with no correctness gate, so this runs the sharded-vs-serial
# full-step parity check at BOTH level counts, on the same mesh, same devices,
# same tolerances. A recommendation to change the model's vertical grid should
# not rest on timings alone.
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_JIT_CACHE_DIR="$SCRATCH/legoesm_jit_cache_j${SLURM_JOB_ID}"
export LEGOESM_XLA_OVERLAP=0
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || exit 2
rc=0
for NLEV in 26 32; do
  echo "=== parity nlev=$NLEV ==="
  LEGOESM_MPAS_WIDE_HALO=0 LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    timeout 900 srun --export=ALL --nodes=1 --ntasks=4 --ntasks-per-node=4 \
      --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 4 --subdivision 5 --nlev "$NLEV" \
      --steps 4 --warmup 1 --lloyd 0 --partition-method metis \
      --halo-strategy ppermute --reorder-for 4 --parity-gate \
    || { echo "PARITY nlev=$NLEV FAILED"; rc=1; }
done
echo "DONE rc=$rc"; exit $rc
