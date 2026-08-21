#!/bin/bash -l
# WHERE DO THE LAT-LON STEP'S TRANSPOSES COME FROM?
#
# Two device traces of the same step at 32 and 128 GPUs showed that every
# class of kernel scales 3.2-3.5x for a 4x device increase EXCEPT transposes,
# which scale 1.38x and account for two thirds of the whole local scaling
# loss. At 128 GPUs they are 31% of local compute, roughly 49 kernels a step.
#
# The model source contains exactly one explicit axis permutation on this
# path, and a CPU dump of the same step's HLO contains ZERO transposes before
# or after optimisation — 288 reshapes, but no transpose. So the transposes
# are inserted by the GPU backend, and this job reads them off the GPU's own
# post-optimisation HLO: how many, on what shapes, with what permutation, and
# what feeds them.
#
# Cheap by construction: two devices, a small grid, three steps. Layout
# assignment is a property of the program and the backend, not of the grid
# size or the device count, so nothing here needs a large allocation. The
# COUNTS are what transfer; the per-kernel times do not, and are not taken.
#SBATCH --job-name=ll_layout
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=ll_layout.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/ll_layout_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR/hlo"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump

# TWO device counts on the SAME grid, so the shard's row count halves while
# every edge-row array keeps its shape. A transpose whose leading extent
# follows the shard is work on owned rows; one that stays at 1 or 2 rows is
# the halo's edge staging, and only the second kind can be device-count
# independent. That is the discriminator, and it costs a minute.
# TWO SHARD SIZES on the same two devices, by changing the GRID rather than
# the device count: a 128-row grid gives each device 64 owned rows, a 256-row
# grid gives 128. Every edge-row array keeps its shape either way. A transpose
# whose leading extent follows the shard is work on owned rows; one that stays
# at 1 or 2 rows is the halo's edge staging, and only the second kind can be
# device-count independent. That is the discriminator.
#
# Single process, two local devices: the multi-controller launcher needs a
# coordination service that does not come up on this partition, and it is not
# needed to make a compiled module.
run_dump () {  # n_lat n_lon tag
  local NLAT=$1 NLON=$2 TAG=$3
  mkdir -p "$OUTDIR/hlo_$TAG"
  echo "=== dump at ${NLAT}x${NLON}, 2 devices (${TAG}) ==="
  LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
  LEGOESM_LATLON_HALO_NOCOMM=1 JAX_ENABLE_X64=0 \
  XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_dump_to=$OUTDIR/hlo_$TAG" \
    timeout 900 srun --export=ALL --nodes=1 --ntasks=1 --gpus-per-node=2 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
      --n-devices 2 --mode strong \
      --n-lat "$NLAT" --n-lon "$NLON" --nlev 26 \
      --steps 3 --warmup 1 --out "$OUTDIR/probe_$TAG.jsonl" \
      || { echo "DUMP $TAG FAILED"; return 1; }
  "$PY" scripts/bench/census_layout_transposes.py --dump-dir "$OUTDIR/hlo_$TAG" \
    --module-re "jit__scan_run" --out "$OUTDIR/census_$TAG.json"
}
rc=0
run_dump 128 256 lat128 || rc=1
run_dump 256 256 lat256 || rc=1
if [ "$rc" -ne 0 ]; then echo "a dump failed; no verdict"; exit 1; fi
echo DONE
