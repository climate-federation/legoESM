#!/bin/bash
#SBATCH --job-name=legoesm_mpi_venv
#SBATCH --output=results/latlon_mpi/setup_venv_%j.out
#SBATCH --error=results/latlon_mpi/setup_venv_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#
# One-time setup: create a pg2328-owned virtualenv that inherits the
# jn2808 conda env's site-packages and adds mpi4jax (which jn2808's env
# lacks).  This is the prerequisite for
# ``scripts/run_latlon_mpi_halo_smoke.sbatch`` (the smoke sbatch has a
# pre-flight that refuses to run without mpi4jax in scope).
#
# After this completes, the new venv at ``~/.venvs/legoesm-mpi/bin/python``
# will be auto-detected by the smoke sbatch.
#
# Usage::
#
#   sbatch scripts/setup_legoesm_mpi_venv.sbatch
#
# Re-running is safe: if the venv already exists it is reused; pip
# install is idempotent.

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/latlon_mpi

# --- Module load: openmpi (provides mpicc + libmpi for mpi4jax build) ---
echo "[$(date -u)] Loading openmpi module …"
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load openmpi/gcc/64/4.1.7a1
echo "    which mpicc → $(which mpicc 2>/dev/null || echo '<missing>')"
echo "    which mpirun → $(which mpirun 2>/dev/null || echo '<missing>')"

# --- Build the venv from jn2808's conda env, inheriting jax/numpy/etc. ---
PY_BASE="/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python"
VENV_DIR="${HOME}/.venvs/legoesm-mpi"

if [[ ! -x "${PY_BASE}" ]]; then
    echo "FATAL: base Python not found at ${PY_BASE}" >&2
    exit 1
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "[$(date -u)] venv already exists at ${VENV_DIR}, reusing"
else
    echo "[$(date -u)] Creating venv at ${VENV_DIR} (inheriting jn2808 site-packages) …"
    mkdir -p "$(dirname "${VENV_DIR}")"
    "${PY_BASE}" -m venv --system-site-packages "${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

echo "[$(date -u)] Venv Python: $(${VENV_PY} --version)"

# --- pip install mpi4jax + cython (mpi4jax build dep) ---
#
# Limit make to 1 thread so the cython build doesn't oversubscribe the
# 4 cores SLURM gave us.  Pin mpi4jax to a version known to work with
# jax 0.9.x (the version in jn2808's env).  The pin can be relaxed
# later — for the smoke test we just need it to import + sendrecv.
echo "[$(date -u)] pip install …"
export MAKEFLAGS="-j1"
"${VENV_PIP}" install --upgrade pip setuptools wheel
"${VENV_PIP}" install cython
# Pin to mpi4jax 0.8.x — legoESM's parallel.reductions warns at 0.9+
# because mpi4jax 0.8 uses the JAX custom-call API that was deprecated
# in JAX 0.9 and removed in JAX 0.10.  Until an FFI-based mpi4jax 0.10
# ships, pin to the tested range to keep MPI execution correct.
"${VENV_PIP}" install --upgrade 'mpi4jax>=0.8,<0.9'

# --- Verify ---
echo
echo "[$(date -u)] Verifying imports in the new venv …"
"${VENV_PY}" -c "
import sys
print(f'Python: {sys.version.splitlines()[0]}')
import jax
print(f'jax {jax.__version__} devices: {jax.devices()}')
import mpi4py, mpi4jax
print(f'mpi4py {mpi4py.__version__}')
print(f'mpi4jax {getattr(mpi4jax, \"__version__\", \"<no __version__>\")}')
import legoesm
print(f'legoesm import OK from {legoesm.__file__}')
print('SETUP OK')
"

echo
echo "[$(date -u)] === Setup complete ==="
echo "    Use:  PY=${VENV_PY}"
echo "    The smoke sbatch auto-detects this path."
