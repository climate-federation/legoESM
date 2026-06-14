#!/usr/bin/env bash
#
# Set up the isolated jax_scm virtual environment used to (re)generate
# the SCM oracle NetCDF outputs under ``tests/validation/scm_oracle/``.
#
# Why a separate venv?
#   jax_scm pins ``jax==0.9.*``; legoESM tracks the newer ``jax`` line.
#   Co-installing both into the same environment would break either the
#   oracle or the production legoESM tests.  The oracle script
#   ``scripts/matrix/run_scm_test_matrix.py oracle`` runs only inside this venv; the
#   benchmarks in ``tests/validation/test_scm_*.py`` load the resulting
#   NetCDF files from disk and don't need jax_scm at import time.
#
# Usage:
#   bash scripts/setup_jax_scm_oracle.sh             # default checkout
#   JAX_SCM_REPO=/path/to/local bash setup_jax_scm_oracle.sh
#

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv-jax-scm}"
JAX_SCM_REPO="${JAX_SCM_REPO:-https://github.com/mpierzyna/jax_scm.git}"
PYTHON="${PYTHON:-.venv/bin/python}"   # use the same interpreter version as legoESM's main venv

if [ ! -x "${PYTHON}" ]; then
    echo "[setup_jax_scm_oracle] cannot find legoESM python at ${PYTHON}; aborting." >&2
    exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup_jax_scm_oracle] creating venv at ${VENV_DIR}"
    "${PYTHON}" -m venv "${VENV_DIR}"
fi

echo "[setup_jax_scm_oracle] upgrading pip in ${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null

echo "[setup_jax_scm_oracle] installing jax_scm from ${JAX_SCM_REPO}"
if [ -d "${JAX_SCM_REPO}" ]; then
    "${VENV_DIR}/bin/pip" install "${JAX_SCM_REPO}"
else
    "${VENV_DIR}/bin/pip" install "git+${JAX_SCM_REPO}"
fi

echo "[setup_jax_scm_oracle] done.  Regenerate oracle outputs with:"
echo "  JAX_PLATFORMS=cpu ${VENV_DIR}/bin/python scripts/matrix/run_scm_test_matrix.py oracle"
