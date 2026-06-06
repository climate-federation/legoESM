#!/usr/bin/env bash
# Set up a working MPI environment for legoESM on a machine that has a
# USER-SPACE MPICH (no system OpenMPI / no sudo needed).
#
# Why a separate venv
# -------------------
# The main ``.venv`` pins JAX 0.10, but ``mpi4jax`` (used by legoESM's halo
# exchange) needs JAX <= 0.9 — JAX 0.10 removed the ``custom_call`` API mpi4jax
# relies on (see ``requirements_mpi.txt``). So MPI runs in a dedicated
# ``.venv-mpi`` with the pinned stack, built against the local MPICH.
#
# Verified on Ubuntu 24.04 with MPICH 4.2.3 at ~/.local/mpich:
#   mpi4jax.sendrecv works under ``mpirun -np 2``; 31 distributed tests pass.
#
# Usage:
#   bash scripts/experiment/setup_mpi_local.sh           # build + smoke-test
#   source scripts/experiment/setup_mpi_local.sh env     # just export the env
set -euo pipefail

# --- locate a user-space MPICH (override MPICH_HOME to point elsewhere) -------
MPICH_HOME="${MPICH_HOME:-$HOME/.local/mpich}"
if [[ ! -x "$MPICH_HOME/bin/mpicc" ]]; then
  echo "ERROR: MPICH not found at $MPICH_HOME (set MPICH_HOME)." >&2
  echo "Build it user-space:  ./configure --prefix=\$HOME/.local/mpich && make -j && make install" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$MPICH_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$MPICH_HOME/lib:${LD_LIBRARY_PATH:-}"
export MPICC="$MPICH_HOME/bin/mpicc"
# legoESM MPI runs CPU + x64; mpi4jax on JAX 0.9 emits a (harmless) deprecation.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"

# ``source ... env`` → only export the environment, do not (re)build.
if [[ "${1:-}" == "env" ]]; then
  echo "MPI env set: $(mpichversion 2>/dev/null | head -1), MPICC=$MPICC"
  return 0 2>/dev/null || exit 0
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
VENV=".venv-mpi"

echo "== creating $VENV =="
uv venv --python 3.12 "$VENV"
echo "== installing the pinned MPI stack (requirements_mpi.txt) =="
uv pip install --python "$VENV/bin/python" -r requirements_mpi.txt
# mpi4py / mpi4jax must be COMPILED against the local MPICH (no binary wheels).
uv pip install --python "$VENV/bin/python" --no-binary mpi4py,mpi4jax \
    --reinstall-package mpi4py --reinstall-package mpi4jax \
    "mpi4py>=4.0,<5.0" "mpi4jax==0.8.1.post2"
echo "== installing legoESM (editable) + pytest, re-pinning jax 0.9.2 =="
uv pip install --python "$VENV/bin/python" -e . pytest pytest-timeout
uv pip install --python "$VENV/bin/python" "jax==0.9.2" "jaxlib==0.9.2"

echo "== smoke: mpi4jax sendrecv under mpirun -np 2 =="
mpirun -np 2 "$VENV/bin/python" -c "
import jax.numpy as jnp, mpi4jax
from mpi4py import MPI
c = MPI.COMM_WORLD; r = c.Get_rank()
y = mpi4jax.sendrecv(jnp.array([float(r)]), jnp.array([0.0]),
                     source=(r+1)%2, dest=(r+1)%2, comm=c)
print(f'rank {r}: recv {float(y[0])}')" 2>&1 | grep -vE "API_VERSION|custom_call|Prefer|removed in"

echo "== distributed tests (np=2) =="
mpirun -np 2 "$VENV/bin/python" -m pytest \
    tests/distributed/test_mpi_bootstrap.py tests/distributed/test_halo_mpi.py \
    -q -p no:cacheprovider 2>&1 | grep -aE "passed|failed|error"

cat <<'DONE'

MPI ready. To run legoESM under MPI:
  source scripts/experiment/setup_mpi_local.sh env
  mpirun -np 2 .venv-mpi/bin/python -m pytest tests/distributed/ -q
DONE
