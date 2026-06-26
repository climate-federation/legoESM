#!/bin/bash -l
#PBS -N cube_fullnode_cpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=128:mpiprocs=6:ompthreads=21
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Cubed-sphere FULL-NODE CPU throughput on NCAR Derecho -- for a fair
# "one full CPU node vs one A100" comparison.  Runs as a BATCH job or
# interactively.
#
# Cubed-sphere has only 6 faces, so --cs-mpi-scatter caps MPI at 6 ranks; you
# CANNOT launch 128 MPI ranks for a cube.  To use the WHOLE 128-core node we run
# a HYBRID layout: 6 MPI ranks (one face each = the genuine domain split) x
# ~21 XLA/Eigen threads per rank, each rank bound to its own core block ->
# ~126 of 128 cores busy.  Reports SYPD per resolution; compare that against
# the 1-A100 number from `RANKS=1 ./cube_strong_gpu.sh` (its n1 rows).
#
# Same driver/flags as the GPU side (run_levante_gpu_scaling.py
# --cs-mpi-scatter); only the backend, conda env, and thread layout differ.
# (NOT run_cpu_mpi_scaling.py --cs-spmd: its cube path is jax.distributed,
# which does not auto-detect Derecho's Cray PALS launcher.)
#
# SUBMIT (qsub from the repo root so $PBS_O_WORKDIR resolves):
#   cd /glade/work/$USER/legoESM
#   qsub scripts/cluster/scaling_derecho/cube_fullnode_cpu.sh
#   # C192 on CPU is slow -- trim for a quick first pass:
#   qsub -v STRONG_RES=48,96 scripts/cluster/scaling_derecho/cube_fullnode_cpu.sh
#
# Or run interactively (qsub -I -l select=1:ncpus=128:mpiprocs=6:ompthreads=21
# ... then ./cube_fullnode_cpu.sh).
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

# Force CPU + the mpi-enabled conda env BEFORE sourcing _env.sh (symmetric with
# the GPU script's cuda pin) so a stale JAX_PLATFORMS can't flip the backend.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-mpi}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# Runtime MPI stack: the SAME GNU cray-mpich mpi4py/mpi4jax were built against.
module load gcc cray-mpich 2>/dev/null || true

# --- Hybrid layout: fill the whole node --------------------------------------
NRANKS="${NRANKS:-6}"              # one MPI rank per cube face (must DIVIDE 6)
NCPUS="${NCPUS:-$(nproc)}"         # cores visible to the job (128 on a full node)
THREADS="${THREADS:-$(( NCPUS / NRANKS ))}"   # XLA/Eigen threads per rank (~21)
# Multi-threaded Eigen ON (do NOT disable it -- that is the 1-core-per-rank mode
# of cube_strong_cpu.sh).  Bound below by --cpu-bind so each rank's THREADS
# threads stay inside its own core block and ranks never oversubscribe.
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- Sweep configuration (overridable via the environment / qsub -v) ---------
PHYSICS="${PHYSICS:-none}"         # 'none' = dycore-only (aggregate case 'dry')
PRECISION="${PRECISION:-float32}"  # match the A100 run
STRONG_RES="${STRONG_RES:-48,96,192}"   # cube face-edge cells (C48/C96/C192)
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cube_cpu_fullnode_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== cube CPU FULL-NODE: ${NRANKS} ranks x ${THREADS} threads = $(( NRANKS * THREADS )) of ${NCPUS} cores ==="
echo "    physics=$PHYSICS prec=$PRECISION res=$STRONG_RES  outdir=$CAMP"

# Hybrid bind: each of the NRANKS ranks pinned to its own depth=THREADS core
# block so their Eigen pools do not collide (Cray PALS syntax; if mpiexec
# rejects these flags on a newer PALS, drop to `--cpu-bind depth -d $THREADS`).
mpiexec -n "$NRANKS" --ppn "$NRANKS" --cpu-bind depth --depth "$THREADS" \
    "$PY" scripts/bench/run_levante_gpu_scaling.py \
    --grid cubed-sphere --cs-mpi-scatter --mode strong \
    --physics "$PHYSICS" --precision "$PRECISION" \
    --strong-resolutions "$STRONG_RES" \
    --output-dir "$CAMP/fullnode" --no-timestamp < /dev/null
rc=$?

# --- Aggregate to a tidy CSV (SYPD per resolution) ---------------------------
"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cube_cpu_fullnode_tidy.csv"

echo "=== DONE rc=$rc ==="
echo "CSV: $CAMP/cube_cpu_fullnode_tidy.csv"
echo "Compare the 'sypd' column (per resolution) against the 1-A100 run:"
echo "  RANKS=1 ./scripts/cluster/scaling_derecho/cube_strong_gpu.sh"
exit $rc
