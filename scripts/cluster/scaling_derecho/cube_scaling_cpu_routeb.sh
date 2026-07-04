#!/bin/bash -l
#PBS -N cube_scaling_cpu_routeb
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=128:mpiprocs=6:ompthreads=21
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Cubed-sphere CPU STRONG-SCALING sweep via ROUTE B (#764 item 1/2):
# jax.distributed multi-controller (gloo on CPU, NCCL on GPU) — the SAME
# route-B stack the latlon sweep uses (bench_atm_latlon_spmd_scaling.py),
# so cube and latlon land as separable curves in one CPU-vs-A100 campaign.
#
# Route A (cube_scaling_cpu.sh) uses mpi4jax face-scatter and caps at 6.
# Route B here uses ``run_cpu_mpi_scaling.py --cs-spmd`` and sweeps the
# FACE-DIVISOR device ladder N = 1,2,3,6, optionally extending past 6 to
# the sub-face tiled counts 6*kt^2 (24, 54) via ``CUBE_SPMD_RANKS`` — the
# P4 sub-face regime.  Each rank is ONE jax.distributed process holding a
# face/tile shard (NOT threads); the node is filled by giving each rank
# ~128/N XLA threads, bound to its own core block (Cray PALS depth).
#
# The cube device ladder + resolution=face-edge differ from the latlon
# lat-band ladder: this is a SEPARATE curve, not points on the latlon
# device axis.  aggregate_bcw_scaling.py keys rows on (grid, n_devices,
# resolution) so the two lanes never overlay (tests/bench/
# test_aggregate_bcw_scaling.py::test_cube_and_latlon_lanes_are_distinct_curves).
#
# Verified on Ginsburg CPU: the np=1 cs-spmd path emits an
# aggregation-ready JSON (grid_type='cubed-sphere', sypd, resolution,
# n_ranks); the multi-rank gloo sweep runs on Derecho/Levante.
#
# Driven by submit_scaling.sh (one job per resolution).  Direct:
#   STRONG_RES=96 ./cube_scaling_cpu_routeb.sh
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
# Face-divisor ladder (each must divide 6).  Default caps at 6 to match the
# PBS `select=...:mpiprocs=6` header — one process per face on one node.
# The P4 sub-face regime (6*kt^2 = 24, 54) needs MORE MPI slots than this
# header grants, so it would oversubscribe: to run it, ALSO raise the PBS
# select (e.g. -l select=1:ncpus=128:mpiprocs=24 for kt=2, or span nodes)
# AND set the ladder:
#   CUBE_SPMD_RANKS="6 24 54" qsub -l select=1:ncpus=128:mpiprocs=54 ...
# A rank count exceeding the granted slots is rejected below rather than
# silently oversubscribed.
CUBE_SPMD_RANKS="${CUBE_SPMD_RANKS:-1 2 3 6}"
_CORES="${LEGOESM_NCPUS:-128}"          # full Derecho node cores
PHYSICS="${PHYSICS:-none}"              # 'none' = dycore-only (aggregate 'dry')
PRECISION="${PRECISION:-float32}"
STRONG_RES="${STRONG_RES:-48 96 192}"   # cube face-edge cells (space-separated)
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cubed-sphere_cpu_routeb_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== cube CPU route-B strong scaling: ranks=[$CUBE_SPMD_RANKS] on ${_CORES} cores  res=[$STRONG_RES] ==="
echo "    physics=$PHYSICS prec=$PRECISION  outdir=$CAMP"

# Granted MPI slots (PBS $PBS_NODEFILE line count, or 1 outside PBS) — a
# ladder rung exceeding this would oversubscribe under Cray PALS, so reject
# it loudly rather than emit meaningless timings.
_SLOTS=1
[ -n "${PBS_NODEFILE:-}" ] && [ -f "${PBS_NODEFILE}" ] \
    && _SLOTS=$(wc -l < "${PBS_NODEFILE}")

rc_all=0
for RES in $STRONG_RES; do
  for N in $CUBE_SPMD_RANKS; do
    if [ -n "${PBS_NODEFILE:-}" ] && [ "$N" -gt "$_SLOTS" ]; then
      echo "  SKIP res=$RES ranks=$N: exceeds granted MPI slots ($_SLOTS) "\
"— raise the PBS select= mpiprocs to run this rung"; rc_all=1; continue
    fi
    THREADS=$(( _CORES / N )); [ "$THREADS" -lt 1 ] && THREADS=1
    # Each rank = one jax.distributed process (a face/tile shard); fill the
    # node with THREADS XLA threads per rank, bound to its own core block.
    export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
           OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
    export OMP_PROC_BIND=close OMP_PLACES=cores
    echo "--- cube route-B ranks=$N x ${THREADS} threads  res=$RES ---"
    # Per (res, N) output subdir so the fixed result filename never collides.
    mpiexec -n "$N" --cpu-bind depth --depth "$THREADS" \
        "$PY" scripts/bench/run_cpu_mpi_scaling.py \
        --grid cubed-sphere --cs-spmd --mode single \
        --resolution "$RES" --physics "$PHYSICS" --precision "$PRECISION" \
        --device cpu --output-dir "$CAMP/r${RES}_n${N}" < /dev/null \
      || { echo "  res=$RES ranks=$N FAILED rc=$?"; rc_all=1; }
  done
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cubed-sphere_cpu_routeb_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/cubed-sphere_cpu_routeb_tidy.csv"
exit $rc_all
