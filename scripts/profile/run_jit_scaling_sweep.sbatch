#!/bin/bash
#SBATCH --job-name=profile_jit_scaling
#SBATCH --output=results/profile_jit_compile/scaling_%j.out
#SBATCH --error=results/profile_jit_compile/scaling_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Task #25 scaling sweep — measure how JIT compile time grows with grid
# size, for both ``ssp_rk3`` (inline 3 stages) and ``ssp_rk3_scan``
# (3 stages folded into a single ``lax.scan`` body).
#
# Why this is interesting:
# At n_lat=16 (run_jit_compile_profile.sbatch) the scan variant
# already delivered ~1.7× compile-time speedup.  The dominant question
# now is whether the speedup GROWS at production scale (n_lat=180).
# XLA optimiser passes scale super-linearly in HLO op count, so a
# fixed 2× reduction in jaxpr equation count at small scale can blow
# up to 5–10× at large scale.  We need data.
#
# This serial sweep (np=1) measures the per-process compile time —
# under MPI each rank's compile is independent so per-rank time is
# what scales.  Larger grids = more HLO ops in the FFT, the gradient
# operators, the polar filter mask shapes, the cell-area arrays, etc.

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/profile_jit_compile

PY="${HOME}/.venvs/legoesm-mpi/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "  JIT compile scaling sweep (Task #25)"
echo "  Job ID: ${SLURM_JOB_ID:-<no-slurm>}"
echo "========================================"

# Sweep across grid sizes.  Production AMIP is n_lat=180.  Bigger
# than n_lat=90 may exhaust the 3-h walltime on inline, so stop the
# inline sweep at 64 if the previous size hit >30 min.
# Use a representative tracer count (3 = q_v/q_c/q_r AMIP default)
# and polar filter on (lifts pole-CFL — the production setup).
for N_LAT in 8 16 32 64; do
    N_LON=$((N_LAT * 2))
    for INT in ssp_rk3 ssp_rk3_scan; do
        echo
        echo "=== n_lat=${N_LAT}, n_lon=${N_LON}, nlev=16, tracers=3, polar=on, integrator=${INT} ==="
        # Allow per-iteration walltime budget — if inline at n_lat=64
        # exceeds 30 min, log the timeout and move on so the sweep
        # produces partial data instead of hitting wall.
        timeout 1800 "${PY}" "${REPO_ROOT}/scripts/profile/profile_step_jit_compile.py" \
            --n-lat "${N_LAT}" --n-lon "${N_LON}" --nlev 16 \
            --tracers 3 --use-polar-filter --integrator "${INT}" \
            || echo "TIMEOUT/FAIL at n_lat=${N_LAT}, integrator=${INT}"
    done
done

echo
echo "========================================"
echo "  Scaling sweep complete"
echo "  Date: $(date -u)"
echo "========================================"
