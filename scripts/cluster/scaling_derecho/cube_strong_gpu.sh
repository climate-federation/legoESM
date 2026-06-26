#!/bin/bash -l
# ===========================================================================
# Interactive cubed-sphere GPU STRONG-scaling sweep on NCAR Derecho.
#
# Runs scripts/bench/run_levante_gpu_scaling.py with TRUE face decomposition
# (--cs-mpi-scatter: each rank owns 6/nranks faces, cross-face halos over
# mpi4jax) at a ladder of rank counts, then aggregates + plots per-resolution
# strong-scaling curves.  This is the GPU half of an apples-to-apples
# CPU-vs-GPU comparison: cube_strong_cpu.sh runs the IDENTICAL driver and flags
# on the CPU backend (only JAX_PLATFORMS + the conda env differ).
#
# RUN inside an interactive GPU job (NOT qsub'd as a batch script):
#   qsub -I -A P08010000 -q main -l walltime=01:00:00 \
#        -l select=1:ncpus=64:mpiprocs=4:ngpus=4:gpu_type=a100
#   cd /glade/work/$USER/legoESM
#   ./scripts/cluster/scaling_derecho/cube_strong_gpu.sh
#
# OVERRIDE knobs from the environment (no edits needed):
#   RANKS="1 2 3 6"  PHYSICS=none  PRECISION=float32  STRONG_RES=48,96,192 \
#   CAMP=$SCRATCH/legoesm_scaling/my_run  ./.../cube_strong_gpu.sh
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Force CUDA + the GPU conda env BEFORE sourcing _env.sh so a stale
# JAX_PLATFORMS=cpu from a reused interactive shell can never silently run this
# GPU sweep on the CPU (symmetric with cube_strong_cpu.sh).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# --- Sweep configuration (all overridable via the environment) ---------------
RANKS="${RANKS:-1 2 3 6}"          # cube face-scatter: rank count must DIVIDE 6
PHYSICS="${PHYSICS:-none}"         # 'none' = dycore-only (aggregate case 'dry')
PRECISION="${PRECISION:-float32}"  # float32 | float64 | both
STRONG_RES="${STRONG_RES:-48,96,192}"   # cube face-edge cells (C48/C96/C192)
STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/cube_gpu_strong_${STAMP}}"
mkdir -p "$CAMP"

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
  mpiexec -n "$N" bash -c "$PIN" _ \
      "$PY" scripts/bench/run_levante_gpu_scaling.py \
      --grid cubed-sphere --cs-mpi-scatter --mode strong \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --strong-resolutions "$STRONG_RES" \
      --output-dir "$CAMP/n$N" --no-timestamp < /dev/null \
    || { echo "  n$N FAILED rc=$?"; rc_all=1; }
done

# --- Aggregate + plot --------------------------------------------------------
"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/cube_gpu_strong_tidy.csv"
"$PY" scripts/plot/plot_strong_scaling_by_resolution.py \
    --csv "$CAMP/cube_gpu_strong_tidy.csv" --grid cubed-sphere --out "$CAMP/plots"

echo "=== DONE rc=$rc_all ==="
echo "CSV:   $CAMP/cube_gpu_strong_tidy.csv"
echo "Plots: $CAMP/plots"
exit $rc_all
