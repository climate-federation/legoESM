#!/bin/bash -l
#PBS -N fullnode_gpu
#PBS -A P08010000
#PBS -q main
#PBS -l job_priority=regular
#PBS -l select=1:ncpus=16:mpiprocs=1:ngpus=1:gpu_type=a100:mem=100GB
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -k eod
# ===========================================================================
# 1-A100 baseline for latlon / icosahedral / spectral -- the GPU half of the
# "one full CPU node vs one A100" comparison.  Pair with fullnode_cpu.sh; both
# use the SAME driver (run_cpu_mpi_scaling.py) and resolutions so the only thing
# that differs is the backend.  Runs as a BATCH job or interactively.
#
# One MPI rank on one A100 (no decomposition, no mpi4jax halo) runs the whole
# problem at each resolution and reports SYPD.  Cubed-sphere is NOT handled here
# (its CPU side uses run_levante --cs-mpi-scatter, so its A100 baseline must too:
# run `RANKS=1 ./cube_strong_gpu.sh` and read the n1 rows).
#
# SUBMIT (qsub from the repo root so $PBS_O_WORKDIR resolves):
#   cd /glade/work/$USER/legoESM
#   qsub -v GRID=latlon       scripts/cluster/scaling_derecho/fullnode_gpu.sh
#   qsub -v GRID=icosahedral  scripts/cluster/scaling_derecho/fullnode_gpu.sh
#   qsub -v GRID=spectral     scripts/cluster/scaling_derecho/fullnode_gpu.sh
#
# Or interactively: GRID=latlon ./fullnode_gpu.sh
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
# JAX_PLATFORMS=cpu can never silently run this baseline on the CPU.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export LEGOESM_CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"
cd "$REPO"

# --- Per-grid resolutions (MUST match fullnode_cpu.sh) -----------------------
GRID="${GRID:-${1:-latlon}}"
PHYSICS="${PHYSICS:-none}"
PRECISION="${PRECISION:-float32}"
# 1 rank = no decomposition, so NO --latlon-2d here (it only engages at >1 rank);
# resolutions MUST match fullnode_cpu.sh so the SYPD points line up.
case "$GRID" in
  cubed-sphere)
    echo "ERROR: cubed-sphere A100 baseline uses run_levante (to match its CPU" >&2
    echo "       side).  Run: RANKS=1 ./cube_strong_gpu.sh  (read the n1 rows)." >&2
    exit 2 ;;
  latlon)       RESOLUTIONS="${RESOLUTIONS:-128 256}" ;;
  icosahedral)  RESOLUTIONS="${RESOLUTIONS:-6 7}" ;;
  spectral)     RESOLUTIONS="${RESOLUTIONS:-85 170}" ;;
  *)
    echo "ERROR: unknown GRID='$GRID' (expected latlon | icosahedral | spectral)" >&2
    exit 2 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
CAMP="${CAMP:-$SCRATCH/legoesm_scaling/${GRID}_a100_${STAMP}}"
mkdir -p "$CAMP"
[ -n "${PBS_O_WORKDIR:-}" ] && exec > >(tee -a "$CAMP/run.log") 2>&1

echo "=== $GRID 1-A100 baseline: physics=$PHYSICS prec=$PRECISION res=[$RESOLUTIONS] ==="
echo "    outdir=$CAMP"

rc_all=0
for R in $RESOLUTIONS; do
  echo "--- $GRID res=$R ---"
  # One rank, one A100 (the PBS allocation exposes a single GPU; no halo, no
  # mpi4jax overlay needed at n_ranks=1).  --device gpu asserts the cuda backend
  # so a CPU fallback can never be recorded as GPU.
  "$PY" scripts/bench/run_cpu_mpi_scaling.py \
      --grid "$GRID" --mode strong --resolution "$R" \
      --physics "$PHYSICS" --precision "$PRECISION" \
      --device gpu \
      --output-dir "$CAMP" < /dev/null \
    || { echo "  res=$R FAILED rc=$?"; rc_all=1; }
done

"$PY" scripts/bench/aggregate_bcw_scaling.py \
    --root "$CAMP" --out "$CAMP/${GRID}_a100_tidy.csv"

echo "=== DONE rc=$rc_all ==="
echo "CSV: $CAMP/${GRID}_a100_tidy.csv"
echo "Compare the 'sypd' column (per resolution) against the full-node CPU run."
exit $rc_all
