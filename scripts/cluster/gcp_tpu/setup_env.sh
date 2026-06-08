#!/usr/bin/env bash
# Set up the legoESM Python environment ON a Cloud TPU VM.
#
# Run this from the repo root on the TPU VM (after scp/clone):
#   cd ~/legoESM && bash scripts/cluster/gcp_tpu/setup_env.sh
#
# Installs the repo (CPU/base deps) into a venv, then overrides jaxlib with
# the TPU build + libtpu.  JAX_VERSION is pinned to the version validated
# locally (0.10.1); override if you intentionally want a different one — but
# keep jax and the TPU jaxlib at the SAME version.
#
# legoESM requires Python >= 3.11, but TPU VM base images often default to an
# older python3 (e.g. 3.10 on Ubuntu 22.04).  If your default python3 is too
# old, point PYTHON at a newer interpreter, e.g.:
#   sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update
#   sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
#   PYTHON=python3.11 bash scripts/cluster/gcp_tpu/setup_env.sh
set -euo pipefail

JAX_VERSION="${JAX_VERSION:-0.10.1}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"
MIN_PY_MINOR=11  # legoESM requires-python = ">=3.11"

# Resolve repo root from this script's location so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

# Fail early with a clear message if the chosen interpreter is too old, rather
# than letting `pip install -e .` fail deep in the run with a cryptic
# "requires a different Python".
if ! "$PYTHON" --version >/dev/null 2>&1; then
  echo "ERROR: interpreter '$PYTHON' not found. Install Python >=3.${MIN_PY_MINOR}" \
       "and re-run with PYTHON=<interpreter> (see header)." >&2
  exit 1
fi
py_minor="$("$PYTHON" -c 'import sys; print(sys.version_info[1])')"
py_major="$("$PYTHON" -c 'import sys; print(sys.version_info[0])')"
echo "Using interpreter: $PYTHON ($("$PYTHON" --version 2>&1))"
if (( py_major < 3 || (py_major == 3 && py_minor < MIN_PY_MINOR) )); then
  echo "ERROR: legoESM needs Python >=3.${MIN_PY_MINOR}, but '$PYTHON' is" \
       "${py_major}.${py_minor}. On Ubuntu 22.04 the default python3 is 3.10;" \
       "install python3.11 and re-run with PYTHON=python3.11 (see header)." >&2
  exit 1
fi

# Treat the venv as usable only if bin/activate actually exists -- a directory
# left behind by a previously failed `python3 -m venv` (e.g. missing the
# python3-venv package) would otherwise make us skip creation and then fail at
# `source`.  Rebuild a partial venv from scratch.
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  if [[ -e "$VENV_DIR" ]]; then
    echo "Removing incomplete venv at $VENV_DIR ..."
    rm -rf "$VENV_DIR"
  fi
  echo "Creating venv at $VENV_DIR with $PYTHON ..."
  if ! "$PYTHON" -m venv "$VENV_DIR"; then
    echo "ERROR: '$PYTHON -m venv' failed. On Debian/Ubuntu install the venv" \
         "package (e.g. 'sudo apt-get install -y python3.11-venv')." >&2
    exit 1
  fi
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel

# Install the repo and its base dependencies (this pulls a CPU jax/jaxlib).
echo "Installing legoESM (editable) ..."
python -m pip install -e .

# Override jax/jaxlib with the TPU build + libtpu.  This MUST come after the
# editable install so the TPU jaxlib wins over the CPU one pulled in above.
echo "Installing jax[tpu]==$JAX_VERSION ..."
python -m pip install -U "jax[tpu]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo
echo "Verifying TPU is visible to JAX ..."
python - <<'PY'
import jax
devs = jax.devices()
print("default backend:", jax.default_backend())
print("device count   :", len(devs))
for d in devs:
    print("  ", d)
assert jax.default_backend() == "tpu", (
    f"expected TPU backend, got {jax.default_backend()!r}"
)
print("OK: TPU backend active with", len(devs), "chips.")
PY

echo
echo "Setup complete. Run a benchmark with:"
echo "  bash scripts/cluster/gcp_tpu/run_bench.sh"