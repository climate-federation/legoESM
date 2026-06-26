#!/bin/bash -l
#PBS -N cube_strong_cpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -l walltime=03:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Cubed-sphere CPU STRONG-scaling sweep on NCAR Derecho.  Runs both as a BATCH
# job and interactively.
#
# CPU half of the apples-to-apples CPU-vs-GPU comparison: runs the SAME driver
# and SAME flags as cube_strong_gpu.sh (scripts/bench/run_levante_gpu_scaling.py
# --cs-mpi-scatter) on the CPU backend.  --cs-mpi-scatter is mpi4jax-based and
# backend-agnostic, so the cube face decomposition exercises the IDENTICAL code
# path -- the only differences are JAX_PLATFORMS, the conda env, and 1-thread-
# per-rank pinning.  (We deliberately do NOT use run_cpu_mpi_scaling.py
# --cs-spmd here: its cube path is jax.distributed, which does not auto-detect
# Derecho's Cray PALS launcher and would refuse to federate -- see the README.)
#
# SUBMIT as a batch job (qsub from the repo root so $PBS_O_WORKDIR finds it):
#   cd /glade/work/$USER/legoESM
#   qsub scripts/cluster/scaling_derecho/cube_strong_cpu.sh
#   # override knobs at submit time (C192 on CPU is slow -- trim for speed):
#   qsub -v PHYSICS=none,PRECISION=float32,STRONG_RES=48,96 \
#        scripts/cluster/scaling_derecho/cube_strong_cpu.sh
#
# Or RUN interactively (qsub -I ... then ./cube_strong_cpu.sh).
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

# Force CPU + the mpi-enabled conda env BEFORE sourcing _env.sh so a stale
# JAX_PLATFORMS=cuda from a reused interactive shell can never push this MPI
# sweep onto the GPU (symmetric with cube_strong_gpu.sh).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-mpi}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# Runtime MPI stack: load the SAME GNU cray-mpich that mpi4py/mpi4jax were built
# against (README Step 2) so mpiexec + libmpi match the build ABI.
module load gcc cray-mpich 2>/dev/null || true

# One thread per rank: each rank pinned to a single core (single-threaded Eigen)
# so N packed ranks never oversubscribe the node -> clean strong-scaling numbers
# (n_resource = n_cores = n_ranks, mirroring 1 GPU/rank on the GPU side).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"

# --- Sweep configuration (all overridable via the environment) ---------------
RANKS="${RANKS:-1 2 3 6}"          # cube face-scatter: rank count must DIVIDE 6
PHYSICS="${PHYSICS:-none}"         # 'none' = dycore-only (aggregate case 'dry')
PRECISION="${PRECISION:-float32}"  # match the GPU run; float64 cube on CPU is slow
STRONG_RES="${STRONG_RES:-48,96,192}"   # cube face-edge cells (C48/C96/C192)
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cube_cpu_strong_${STAMP}}"
mkdir -p "$CAMP"
# Batch run: mirror console output into the results dir so a log lands next to
# the data regardless of where PBS routes the job's .o file.
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== cube CPU strong sweep: ranks=[$RANKS] physics=$PHYSICS prec=$PRECISION res=$STRONG_RES ==="
echo "    outdir=$CAMP"

rc_all=0
for N in $RANKS; do
  echo "=== $N rank(s) ==="
  # No GPU pin on CPU (run_levante's affinity helper is a harmless no-op here).
  mpiexec -n "$N" \
      "$PY" scripts/bench/run_levante_gpu_scaling.py \
      --grid cubed-sphere --cs-mpi-scatter --mode strong \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --strong-resolutions "$STRONG_RES" \
      --output-dir "$CAMP/n$N" --no-timestamp < /dev/null \
    || { echo "  n$N FAILED rc=$?"; rc_all=1; }
done

# --- Aggregate + plot --------------------------------------------------------
"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cube_cpu_strong_tidy.csv"
"$PY" scripts/plot/plot_strong_scaling_by_resolution.py \
    --csv "$CAMP/cube_cpu_strong_tidy.csv" --grid cubed-sphere --out "$CAMP/plots"

echo "=== DONE rc=$rc_all ==="
echo "CSV:   $CAMP/cube_cpu_strong_tidy.csv"
echo "Plots: $CAMP/plots"
exit $rc_all
