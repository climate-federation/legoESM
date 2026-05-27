#!/bin/bash
#SBATCH --job-name=latlon_mpi_halo_smoke
#SBATCH --output=results/latlon_mpi/halo_smoke_%j.out
#SBATCH --error=results/latlon_mpi/halo_smoke_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Stage-0 smoke test for the lat-lon band-MPI halo machinery introduced in
# ``src/legoesm/parallel/latlon_mpi.py``.  Validates that:
#
#   1. The single-rank branch reproduces serial pad_halo_latlon exactly
#      (pytest tests/parallel/test_latlon_mpi_halo_serial.py).
#   2. The multi-rank MPI exchange matches the same serial reference on
#      the interior lon columns at np=2 and np=4 (pytest under mpirun).
#   3. Reverse-mode AD through the MPI sendrecv returns finite gradients.
#
# These tests gate the Stage-1 implementation of ``make_latlon_mpi_step``:
# they MUST pass before we wire the lat-lon dycore into the distributed
# driver.
#
# Usage::
#
#   sbatch scripts/run_latlon_mpi_halo_smoke.sbatch
#
# Watch with ``squeue -u $USER`` and inspect
# ``results/latlon_mpi/halo_smoke_<jobid>.out`` once the job runs.

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/latlon_mpi

# Prefer the pg2328-owned venv with mpi4jax (created by
# ``scripts/setup_legoesm_mpi_venv.sbatch``); fall back to jn2808's
# conda env when running tests that don't need MPI.  The pre-flight
# below will abort if mpi4jax isn't importable from the chosen Python.
PG_VENV="${HOME}/.venvs/legoesm-mpi/bin/python"
JN_BASE="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
if [[ -x "${PG_VENV}" ]]; then
    PY="${PG_VENV}"
else
    PY="${JN_BASE}"
fi
echo "Using Python: ${PY}"

# Load openmpi (mpirun + libmpi for mpi4jax) on the compute node.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load openmpi/gcc/64/4.1.7a1 2>/dev/null || true

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

# Quiet the JAX threading sprawl so per-rank threads don't oversubscribe
# the cores the SLURM allocation gave us.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "========================================"
echo "  Lat-lon band-MPI halo smoke test"
echo "  Job ID:   ${SLURM_JOB_ID:-<no-slurm>}"
echo "  Date:     $(date -u)"
echo "  Repo:     ${REPO_ROOT}"
echo "  Env:      $(${PY} --version)"
echo "========================================"

# ----------------------------------------------------------------------
# Pre-flight: mpi4jax MUST be importable, otherwise pytest.importorskip
# would silently *pass* every MPI test (zero tests run, exit 0) and we
# would think the halo machinery works when it actually never ran.
# Fail loudly here rather than burn the SLURM allocation on a no-op.
# ----------------------------------------------------------------------
echo
echo "--- Pre-flight: verifying MPI stack is importable ---"
if ! ${PY} -c "import mpi4jax, mpi4py" 2>/dev/null; then
    echo "FATAL: mpi4jax / mpi4py not importable from ${PY}"
    echo
    echo "The jn2808 conda env at /burg-archive/glab/users/jn2808/.conda/envs/legoesm"
    echo "does NOT ship mpi4jax (verified 2026-05).  Build a pg2328-owned"
    echo "virtualenv that inherits from it and pip-installs the MPI stack::"
    echo
    echo "    python -m venv --system-site-packages ~/.venvs/legoesm-mpi"
    echo "    ~/.venvs/legoesm-mpi/bin/pip install mpi4jax cython"
    echo "    # then re-point PY in this sbatch to ~/.venvs/legoesm-mpi/bin/python"
    echo
    echo "Without this fix the MPI tests would silently SKIP and the smoke"
    echo "test would report success on zero exercised tests — defeating the"
    echo "whole point of the gate.  Refusing to submit."
    exit 2
fi
echo "    mpi4jax / mpi4py present — OK"

echo
echo "--- Step 1/3 : serial (n_ranks=1) tests, no MPI ---"
${PY} -m pytest -xvs \
    tests/parallel/test_latlon_mpi_halo_serial.py \
    tests/parallel/test_latlon_mpi_step_serial.py \
    tests/atmosphere/test_cgrid_latlon_pole_v_bc.py

# Detect a usable MPI launcher.  Ginsburg ships srun (preferred under
# SLURM) and OpenMPI's mpirun.  Pick srun when inside SLURM, fall back
# to mpirun for local debugging.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null; then
    MPI_LAUNCH="srun --mpi=pmix --cpus-per-task=1"
else
    MPI_LAUNCH="mpirun"
fi

for NP in 2 4; do
    echo
    echo "--- Step ${NP}-rank : MPI halo tests ---"
    ${MPI_LAUNCH} -n ${NP} ${PY} -m pytest -xvs \
        tests/distributed/test_latlon_mpi_halo.py

    echo
    echo "--- Step ${NP}-rank : full step equivalence ---"
    ${MPI_LAUNCH} -n ${NP} ${PY} -m pytest -xvs \
        tests/distributed/test_latlon_mpi_step.py
done

echo
echo "========================================"
echo "  All Stage-0 smoke tests PASSED"
echo "  Date: $(date -u)"
echo "========================================"
