#!/bin/bash -l
# DOES THE ICOSAHEDRAL LANE FLIP MEMORY ORDER THE WAY THE LAT-LON ONE DOES?
#
# On the lat-lon lane the compiled step held 28 transposes, none of them free,
# converting between (level, lat, lon) and (lat, lon, level), and 26 of them
# acted on halo edge slices whose shape does not follow the shard -- which is
# why that cost is 87% device-count independent. Serialising those payloads
# level-first removed 26 of the 28.
#
# The icosahedral lane stores its state (cells, level) rather than
# (lat, lon, level) and stages its halo through a gather/concatenate/scatter
# on an unstructured neighbour table, so there is no reason to assume the same
# defect and no reason to assume its absence. This job asks, on the GPU's own
# post-optimisation HLO, for the same three numbers: how many transposes, on
# what shapes, and how many of them the compiler made free.
#
# Two subdivisions at fixed device count, so a transpose whose leading extent
# follows the shard is separable from one that does not -- the discriminator
# that settled the lat-lon lane.
#
# Cheap by construction: two devices, small meshes, three steps. Layout
# assignment is a property of the program and the backend, not of the mesh
# size or the device count. The COUNTS transfer; per-kernel times do not, and
# are not taken here.
#SBATCH --job-name=mpas_layout
#SBATCH --account=bd1179_gpu
#SBATCH --partition=gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=mpas_layout.%j.log
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="$SCRATCH/legoesm_scaling/mpas_layout_j${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
export LEGOESM_JIT_CACHE_DIR=""   # a cached compile does not re-dump

rc=0
run_dump () {  # subdivision tag
  local SUB=$1 TAG=$2
  mkdir -p "$OUTDIR/hlo_$TAG"
  echo "=== dump at subdivision ${SUB}, 2 devices (${TAG}) ==="
  LEGOESM_MPAS_WIDE_HALO=1 LEGOESM_MPAS_RAGGED_HALO=0 \
  LEGOESM_MPAS_HALO_BALLAST=1 LEGOESM_MPAS_HALO_NOCOMM=0 \
  LEGOESM_MPAS_HALO_NOSTAGE=0 JAX_ENABLE_X64=0 \
  XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_dump_to=$OUTDIR/hlo_$TAG" \
    timeout --kill-after=30 900 srun --export=ALL --nodes=1 --ntasks=1 \
      --gpus-per-node=2 --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --n-devices 2 --subdivision "$SUB" --nlev 26 \
      --steps 3 --warmup 1 --lloyd 0 --partition-method sfc \
      --out "$OUTDIR/probe_$TAG.jsonl" || { echo "DUMP $TAG FAILED"; return 1; }
  # The lat-lon step's module is jit__scan_run; this lane's is jit__step.
  # Let the census say what it found rather than guessing the name here.
  "$PY" scripts/bench/census_layout_transposes.py --dump-dir "$OUTDIR/hlo_$TAG" \
    --module-re "jit_+step" --out "$OUTDIR/census_$TAG.json"
}
run_dump 5 s5 || rc=1
run_dump 6 s6 || rc=1
if [ "$rc" -ne 0 ]; then echo "a dump failed; no verdict"; exit 1; fi
echo DONE
