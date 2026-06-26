#!/bin/bash -l
#PBS -N fullnode_gpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# GPU STRONG-SCALING sweep for latlon / icosahedral / spectral -- the GPU half
# of the CPU-vs-A100 comparison.  Sweeps 1->2->4 A100 (one rank per GPU, route-A
# mpi4jax) AT EACH resolution, so the GPU side is a real scaling curve up to the
# full GPU node -- not a single A100.  Spectral has no MPI path (1 GPU only).
# Pair with fullnode_cpu.sh; SAME driver (run_cpu_mpi_scaling.py) + resolutions.
#
# REQUIRES the route-A overlay env (legoesm-gpu with a CUDA-built mpi4jax; see
# README Step 1b) -- multi-GPU mpi4jax halos.  Cubed-sphere is handled by
# cube_strong_gpu.sh (face-scatter, RANKS=1 2 3).
#
# Driven by submit_fullnode.sh (one job per resolution).  Direct:
#   GRID=latlon RESOLUTIONS=256 ./fullnode_gpu.sh
# ===========================================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ] \
        && [ -f "${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho/_env.sh" ]; then
    SCRIPT_DIR="${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# Route-A runtime stack (README Step 1b): GNU cray-mpich the bindings were built
# against, the CUDA GTL, GPU-aware MPI, and the Cray libmpi loader bridge.
module load gcc cray-mpich cuda craype-accel-nvidia80 2>/dev/null || true
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# --- Per-grid sweep (raise on cube / unknown: dispatch hardening) ------------
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"
PRECISION="${PRECISION:-float32}"
EXTRA=""
NGPUS_DETECT="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 0)"
NGPUS="${NGPUS:-${NGPUS_DETECT}}"
if ! [ "$NGPUS" -ge 1 ] 2>/dev/null; then NGPUS=4; fi
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere GPU uses cube_strong_gpu.sh (face-scatter)." >&2
    exit 2 ;;
  latlon)
    GPU_RANKS="${GPU_RANKS:-1 2 4}"          # route-A, one rank per A100
    EXTRA="--latlon-2d"
    RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)
    GPU_RANKS="${GPU_RANKS:-1 2 4}"
    RESOLUTIONS="${RESOLUTIONS:-6 7}" ;;
  spectral)
    GPU_RANKS="1"                            # no MPI -> 1 GPU only
    RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (latlon | icosahedral | spectral)" >&2
    exit 2 ;;
esac

# rank -> local GPU pin (Cray PALS; run_cpu_mpi_scaling auto-pin does not read
# PALS, so without this every rank grabs GPU 0 -> the eff=0.5 self-blind bug).
PIN='export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}; exec "$@"'

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_gpu_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== $GRID GPU strong scaling: gpus=[$GPU_RANKS] (NGPUS=$NGPUS) res=[$RESOLUTIONS] ==="
echo "    physics=$PHYSICS prec=$PRECISION  outdir=$CAMP"

rc_all=0
for R in $RESOLUTIONS; do
  for N in $GPU_RANKS; do
    if [ "$N" -gt "$NGPUS" ]; then
      echo "--- $GRID res=$R N=$N > NGPUS=$NGPUS -- SKIP (single node) ---"
      continue
    fi
    echo "--- $GRID res=$R gpus=$N ---"
    mpiexec -n "$N" bash -c "$PIN" _ \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py \
        --grid "$GRID" --mode strong --resolution "$R" \
        --physics "$PHYSICS" --precision "$PRECISION" \
        --device gpu $EXTRA \
        --output-dir "$CAMP" < /dev/null \
      || { echo "  res=$R gpus=$N FAILED rc=$?"; rc_all=1; }
  done
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_gpu_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/${GRID}_gpu_tidy.csv"
exit $rc_all
