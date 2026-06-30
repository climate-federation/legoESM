#!/bin/bash -l
#PBS -N cube_scaling_cpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=128:mpiprocs=6:ompthreads=21
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Cubed-sphere CPU STRONG-SCALING sweep on a full Derecho node -- the cube CPU
# half of the CPU-vs-A100 comparison.  Runs as a BATCH job or interactively.
#
# Cube has only 6 faces, so MPI caps at 6 ranks.  We sweep the FACE
# decomposition N = 1,2,3,6 (matching the cube GPU sweep 1,2,3 A100), and at
# EACH N keep the WHOLE 128-core node busy by giving each rank ~128/N XLA
# threads (1 rank x128, 2x64, 3x42, 6x21), bound to its own core block.  So
# every point is a full-node measurement at a different face split -> a
# node-level strong-scaling curve, not a 1-6-core toy.  Driver:
# run_levante_gpu_scaling.py --cs-mpi-scatter (mpi4jax face scatter, PALS-safe).
#
# Driven by submit_scaling.sh (one job per resolution).  Direct:
#   STRONG_RES=96 ./cube_scaling_cpu.sh
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
module load gcc cray-mpich 2>/dev/null || true

# --- Sweep configuration -----------------------------------------------------
CUBE_RANKS="${CUBE_RANKS:-1 2 3 6}"   # face decomposition (each must DIVIDE 6)
# Total cores to spread as threads across the face ranks.  Do NOT use $NCPUS:
# PBS EXPORTS its own $NCPUS (cores-per-rank, e.g. ompthreads), which is NOT the
# node core count we need.  cube requests mpiprocs=6 so $PBS_NODEFILE counts
# ranks (6), not cores either -- default to a full Derecho node; override with
# $LEGOESM_NCPUS.
_CORES="${LEGOESM_NCPUS:-128}"        # full Derecho node cores
PHYSICS="${PHYSICS:-none}"           # 'none' = dycore-only (aggregate case 'dry')
PRECISION="${PRECISION:-float32}"
STRONG_RES="${STRONG_RES:-48 96 192}"   # cube face-edge cells (space-separated)
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cubed-sphere_cpu_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

# run_levante sweeps resolutions internally, so loop N (face count) only and
# pass all resolutions at once; comma-separate them for --strong-resolutions.
RES_CSV="$(echo "$STRONG_RES" | tr ' ' ',')"
echo "=== cube CPU strong scaling: faces=[$CUBE_RANKS] on ${_CORES} cores  res=[$STRONG_RES] ==="
echo "    physics=$PHYSICS prec=$PRECISION  outdir=$CAMP"

rc_all=0
for N in $CUBE_RANKS; do
  THREADS=$(( _CORES / N ))             # fill the node at this face count
  # Multi-threaded Eigen ON; bind each rank to its own THREADS-core block so the
  # per-rank pools never overlap (Cray PALS depth binding).  Per-N output subdir
  # so run_levante's fixed strong_scaling.json filename never collides.
  export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
         OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
  export OMP_PROC_BIND=close OMP_PLACES=cores
  echo "--- cube faces=$N x ${THREADS} threads ---"
  mpiexec -n "$N" --cpu-bind depth --depth "$THREADS" \
      "$PY" scripts/bench/run_levante_gpu_scaling.py \
      --grid cubed-sphere --cs-mpi-scatter --mode strong \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --strong-resolutions "$RES_CSV" \
      --output-dir "$CAMP/f$N" --no-timestamp < /dev/null \
    || { echo "  faces=$N FAILED rc=$?"; rc_all=1; }
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cubed-sphere_cpu_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/cubed-sphere_cpu_tidy.csv"
exit $rc_all
