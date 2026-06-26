#!/bin/bash -l
#PBS -N fullnode_cpu
#PBS -A P08010000
#PBS -q main
# NB: the queue is overridden and charging set by submit_fullnode.sh -- -q main
#     adds -l job_priority, -q develop omits it (develop takes no job_priority).
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# CPU STRONG-SCALING sweep for latlon / icosahedral / spectral -- the CPU half
# of the CPU-vs-A100 comparison.  Runs as a BATCH job or interactively.  Pair
# with fullnode_gpu.sh (the GPU strong-scaling sweep).
#
# latlon and icosahedral are genuinely domain-decomposed, so we sweep the MPI
# rank ladder 1,2,4,...,128 (one rank per core, true decomposition) AT EACH
# resolution -> a real strong-scaling curve up to the full node.  Spectral has
# no MPI path (rank-1 only); it fills the node with XLA threads as a single
# point.  Driver: run_cpu_mpi_scaling.py (mpi4jax, PALS-safe).  Cubed-sphere is
# handled by cube_fullnode_cpu.sh (face-scatter).
#
# Driven by submit_fullnode.sh (one job per resolution).  Direct:
#   GRID=latlon RESOLUTIONS=256 ./fullnode_cpu.sh
#   qsub -v GRID=icosahedral,RESOLUTIONS=7 fullnode_cpu.sh
# ===========================================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ] \
        && [ -f "${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho/_env.sh" ]; then
    SCRIPT_DIR="${PBS_O_WORKDIR}/scripts/cluster/scaling_derecho"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-mpi}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"
module load gcc cray-mpich 2>/dev/null || true   # match the mpi4py/mpi4jax build ABI

# --- Per-grid sweep (raise on cube / unknown: dispatch hardening) ------------
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"
PRECISION="${PRECISION:-float32}"
NCPUS="${NCPUS:-$(nproc)}"
EXTRA=""
THREADS=1                              # pure-MPI default: one thread per rank
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere uses cube_fullnode_cpu.sh (6-face scatter)." >&2
    exit 2 ;;
  latlon)
    RANKS="${RANKS:-1 2 4 8 16 32 64 128}"   # rank ladder up to the full node
    EXTRA="--latlon-2d"                       # 2-D pencil: low halo at high ranks
    RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)
    RANKS="${RANKS:-1 2 4 8 16 32 64 128}"   # MPAS partitions; powers of 2
    RESOLUTIONS="${RESOLUTIONS:-6 7}" ;;
  spectral)
    RANKS="1"                                 # no MPI -> single point...
    THREADS="${THREADS_SPECTRAL:-$NCPUS}"     # ...fill the node with XLA threads
    RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (latlon | icosahedral | spectral)" >&2
    exit 2 ;;
esac

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
if [ "$THREADS" -le 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_cpu_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== $GRID CPU strong scaling: ranks=[$RANKS] x ${THREADS} thr  res=[$RESOLUTIONS] ==="
echo "    physics=$PHYSICS prec=$PRECISION ncpus=$NCPUS  outdir=$CAMP"

rc_all=0
for R in $RESOLUTIONS; do
  for N in $RANKS; do
    if [ "$(( N * THREADS ))" -gt "$NCPUS" ]; then
      echo "--- $GRID res=$R N=$N: needs $(( N * THREADS )) > $NCPUS cores -- SKIP ---"
      continue
    fi
    echo "--- $GRID res=$R ranks=$N ---"
    mpiexec -n "$N" \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py \
        --grid "$GRID" --mode strong --resolution "$R" \
        --physics "$PHYSICS" --precision "$PRECISION" \
        --device cpu $EXTRA \
        --output-dir "$CAMP" < /dev/null \
      || { echo "  res=$R ranks=$N FAILED rc=$?"; rc_all=1; }
  done
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_cpu_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/${GRID}_cpu_tidy.csv"
exit $rc_all
