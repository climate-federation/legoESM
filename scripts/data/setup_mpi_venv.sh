#!/usr/bin/env bash
# Build a dedicated venv for MPI work using the requirements_mpi.txt
# pinned stack. Resolves F9 (mpi4jax / JAX version mismatch in the
# default .venv that causes ~70x per-step slowdown on multi-rank).
#
# Usage:
#   scripts/setup_mpi_venv.sh             # creates .venv-mpi if missing
#   FORCE=1 scripts/setup_mpi_venv.sh     # rebuild from scratch
#
# Env vars:
#   PYBIN     python interpreter used to bootstrap the venv
#             (default: python3.13)
#   VENV_DIR  target venv directory (default: .venv-mpi)
#
# After this script succeeds, MPI benches/scripts should be invoked
# via the venv binary, e.g.
#   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 mpirun -np 4 \
#       .venv-mpi/bin/python scripts/bench/bench_plane_crm_dd_scaling.py ...
#
# A sanity check at the end runs ``python -c "import jax, mpi4jax;
# print(jax.__version__, mpi4jax.__version__)"`` to verify the pin
# took effect.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYBIN="${PYBIN:-python3.13}"
VENV_DIR="${VENV_DIR:-.venv-mpi}"
FORCE="${FORCE:-0}"

if [ -d "$VENV_DIR" ] && [ "$FORCE" != "1" ]; then
    echo "$VENV_DIR already exists. Set FORCE=1 to rebuild from scratch."
    exit 0
fi

if [ "$FORCE" = "1" ] && [ -d "$VENV_DIR" ]; then
    echo "Removing existing $VENV_DIR"
    rm -rf "$VENV_DIR"
fi

echo "Bootstrapping $VENV_DIR via $PYBIN ..."
"$PYBIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip wheel

echo "Installing MPI-compatible pins from requirements_mpi.txt (first, so the"
echo "federation install below resolves against the pinned jax stack)..."
"$VENV_DIR/bin/pip" install -r requirements_mpi.txt

echo "Installing legoESM federation (editable). Plain 'pip install -e .' cannot"
echo "resolve the uv-workspace members (legoesm-atmosphere etc. are not on"
echo "PyPI); install_federation.py resolves the member DAG locally."
"$VENV_DIR/bin/python" scripts/experiment/install_federation.py --all

echo ""
echo "=== Sanity check ==="
"$VENV_DIR/bin/python" -c "
import jax, mpi4jax, mpi4py
print(f'jax     : {jax.__version__}')
print(f'mpi4jax : {mpi4jax.__version__}')
print(f'mpi4py  : {mpi4py.__version__}')
"

echo ""
echo "OK. Use $VENV_DIR/bin/python for MPI workloads, e.g.:"
echo "  mpirun -np 4 $VENV_DIR/bin/python scripts/bench/bench_plane_crm_dd_scaling.py --mode strong --time-steps 10"
