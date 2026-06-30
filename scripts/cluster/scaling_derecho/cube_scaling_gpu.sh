#!/bin/bash -l
#PBS -N cube_scaling_gpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100:mem=400GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# Cubed-sphere GPU STRONG-scaling sweep on NCAR Derecho.  Runs both as a BATCH
# job and interactively.
#
# Runs scripts/bench/run_levante_gpu_scaling.py with TRUE face decomposition
# (--cs-mpi-scatter: each rank owns 6/nranks faces, cross-face halos over
# mpi4jax) over 1->2->3 A100 (4 doesn't divide 6, 6 needs 2 nodes) -> the cube
# GPU strong-scaling curve.  cube_scaling_cpu.sh is the CPU twin (identical
# driver/flags; only JAX_PLATFORMS + the conda env + thread layout differ).
# Multi-GPU here uses route-A mpi4jax, so it needs the README Step 1b overlay.
#
# SUBMIT as a batch job (qsub from the repo root so $PBS_O_WORKDIR finds it):
#   cd /glade/work/$USER/legoESM
#   qsub scripts/cluster/scaling_derecho/cube_scaling_gpu.sh
#   # override knobs at submit time (no file edits):
#   qsub -v PHYSICS=moist,PRECISION=both,STRONG_RES=48,96 \
#        scripts/cluster/scaling_derecho/cube_scaling_gpu.sh
#
# Or RUN interactively (qsub -I ... then ./cube_scaling_gpu.sh).
#
# NOTE: a single Derecho GPU node = 4 A100; cube face-scatter ranks must DIVIDE
#   6, so one node tops out at RANKS="1 2 3" (4 is invalid, 6 needs 2 nodes:
#   bump the select= line to select=2:...:ngpus=3 and RANKS="...6").
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

# Force CUDA + the GPU conda env BEFORE sourcing _env.sh so a stale
# JAX_PLATFORMS=cpu from a reused interactive shell can never silently run this
# GPU sweep on the CPU (symmetric with cube_scaling_cpu.sh).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# Route-A runtime stack (README Step 1b): multi-GPU mpi4jax face halos need the
# GNU cray-mpich the bindings were built against, the CUDA GTL, GPU-aware MPI,
# and the Cray libmpi loader bridge.  (Inert/harmless at RANKS=1.)
module load gcc cray-mpich cuda craype-accel-nvidia80 2>/dev/null || true
export MPICH_GPU_SUPPORT_ENABLED=1
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

# --- Sweep configuration (all overridable via the environment) ---------------
RANKS="${RANKS:-1 2 3}"            # cube face-scatter: must DIVIDE 6; 1 node=4 GPUs
PHYSICS="${PHYSICS:-none}"         # 'none' = dycore-only (aggregate case 'dry')
PRECISION="${PRECISION:-float32}"  # float32 | float64 | both
STRONG_RES="${STRONG_RES:-48 96 192}"   # cube face-edge cells (space or comma)
RES_CSV="$(echo "$STRONG_RES" | tr ' ' ',')"   # run_levante wants comma-separated
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cube_gpu_strong_${STAMP}}"
mkdir -p "$CAMP"
# Batch run: mirror console output into the results dir so a log lands next to
# the data regardless of where PBS routes the job's .o file.
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

# rank -> local GPU pin.  run_levante's own affinity helper reads OpenMPI/
# MVAPICH/SLURM vars but NOT Cray PALS (Derecho's mpiexec), so without this
# every rank grabs GPU 0 (the documented eff=0.5 self-blinding bug).  The
# script RESPECTS an explicit CUDA_VISIBLE_DEVICES, so this binds rank k -> GPU
# k; the fallbacks keep it portable to an OpenMPI/MVAPICH launcher.
PIN='export CUDA_VISIBLE_DEVICES=${PALS_LOCAL_RANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-0}}}; exec "$@"'

echo "=== cube GPU strong sweep: ranks=[$RANKS] physics=$PHYSICS prec=$PRECISION res=$STRONG_RES ==="
echo "    outdir=$CAMP"

rc_all=0
for N in $RANKS; do
  echo "=== $N GPU(s) ==="
  # Per-N output subdir so run_levante's fixed strong_scaling.json never collides.
  mpiexec -n "$N" bash -c "$PIN" _ \
      "$PY" scripts/bench/run_levante_gpu_scaling.py \
      --grid cubed-sphere --cs-mpi-scatter --mode strong \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --strong-resolutions "$RES_CSV" \
      --output-dir "$CAMP/n$N" --no-timestamp < /dev/null \
    || { echo "  n$N FAILED rc=$?"; rc_all=1; }
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cube_gpu_tidy.csv"

echo "=== DONE rc=$rc_all ===   CSV: $CAMP/cube_gpu_tidy.csv"
exit $rc_all
