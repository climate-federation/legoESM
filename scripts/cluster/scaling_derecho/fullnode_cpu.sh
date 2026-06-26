#!/bin/bash -l
#PBS -N fullnode_cpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# FULL-NODE CPU throughput for latlon / icosahedral / spectral -- the CPU half
# of a "one full CPU node vs one A100" comparison.  Runs as a BATCH job or
# interactively.  Pair with fullnode_gpu.sh (the 1-A100 baseline, SAME driver).
#
# Unlike cubed-sphere (6 faces -> MPI capped at 6 ranks, see cube_fullnode_cpu.sh),
# latlon and icosahedral are genuinely domain-decomposed, so the fair full-node
# test is PURE MPI: one rank per core (128 ranks x 1 thread), true decomposition
# across the whole node -- no thread-fill hybrid needed.  Spectral has no MPI
# path (rank-1 only), so it fills the node with XLA threads instead.
#
# Driver: scripts/bench/run_cpu_mpi_scaling.py (native latlon-band / 2-D-pencil
# and icosahedral-MPAS decomposition over mpi4jax; PALS-safe).  Cubed-sphere is
# NOT handled here -- it needs run_levante --cs-mpi-scatter (cube_fullnode_cpu.sh).
#
# SUBMIT (qsub from the repo root so $PBS_O_WORKDIR resolves):
#   cd /glade/work/$USER/legoESM
#   qsub -v GRID=latlon       scripts/cluster/scaling_derecho/fullnode_cpu.sh
#   qsub -v GRID=icosahedral  scripts/cluster/scaling_derecho/fullnode_cpu.sh
#   qsub -v GRID=spectral     scripts/cluster/scaling_derecho/fullnode_cpu.sh
#   # override resolutions / rank count:
#   qsub -v GRID=latlon,RESOLUTIONS="128 256",NRANKS=64 scripts/.../fullnode_cpu.sh
#
# Or interactively: GRID=latlon ./fullnode_cpu.sh
# ===========================================================================
set -uo pipefail

# Locate _env.sh: under PBS the script runs from a spool copy, so resolve from
# $PBS_O_WORKDIR (the submit dir); fall back to BASH_SOURCE for interactive use.
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

# --- Per-grid full-node layout (raise on cube / unknown: dispatch hardening) --
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"          # none | held_suarez | moist
PRECISION="${PRECISION:-float32}"   # match the A100 run
NCPUS="${NCPUS:-$(nproc)}"          # cores in the node (128 on a full Derecho node)
EXTRA=""
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere uses the dedicated 6-face hybrid script (MPI caps" >&2
    echo "       at 6 ranks).  Run: ./cube_fullnode_cpu.sh" >&2
    exit 2 ;;
  latlon)
    NRANKS="${NRANKS:-$NCPUS}"            # one rank per core; 2-D pencil decomposition
    THREADS=1
    EXTRA="--latlon-2d"                   # low per-rank halo -> 128 ranks at modest res
    RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)
    NRANKS="${NRANKS:-$NCPUS}"            # MPAS partitions; powers of 2 (128 = 2^7)
    THREADS=1
    RESOLUTIONS="${RESOLUTIONS:-6 7}" ;;
  spectral)
    NRANKS=1                              # spectral has NO MPI decomposition
    THREADS="${THREADS:-$NCPUS}"          # fill the node with XLA threads only
    RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (expected latlon | icosahedral | spectral;" >&2
    echo "       cubed-sphere has its own cube_fullnode_cpu.sh)" >&2
    exit 2 ;;
esac

export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
       OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
# 1 thread/rank (pure MPI) -> single-threaded Eigen so packed ranks never
# oversubscribe; many threads (spectral) -> leave Eigen multi-threaded.
if [ "$THREADS" -le 1 ]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_cpu_fullnode_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== $GRID FULL-NODE CPU: ${NRANKS} ranks x ${THREADS} thread(s) of ${NCPUS} cores ==="
echo "    physics=$PHYSICS prec=$PRECISION res=[$RESOLUTIONS]  outdir=$CAMP"

rc_all=0
for R in $RESOLUTIONS; do
  echo "--- $GRID res=$R ---"
  mpiexec -n "$NRANKS" \
      "$PY" scripts/bench/run_cpu_mpi_scaling.py \
      --grid "$GRID" --mode strong --resolution "$R" \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --device cpu $EXTRA \
      --output-dir "$CAMP" < /dev/null \
    || { echo "  res=$R FAILED rc=$?"; rc_all=1; }
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_cpu_fullnode_tidy.csv"

echo "=== DONE rc=$rc_all ==="
echo "CSV: $CAMP/${GRID}_cpu_fullnode_tidy.csv"
echo "Compare the 'sypd' column (per resolution) against the 1-A100 run:"
echo "  GRID=$GRID ./scripts/cluster/scaling_derecho/fullnode_gpu.sh"
exit $rc_all
