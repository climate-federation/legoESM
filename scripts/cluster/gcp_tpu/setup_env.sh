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
set -euo pipefail

JAX_VERSION="${JAX_VERSION:-0.10.1}"
VENV_DIR="${VENV_DIR:-.venv}"

# Resolve repo root from this script's location so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
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