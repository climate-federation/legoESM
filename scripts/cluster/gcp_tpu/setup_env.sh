#!/usr/bin/env bash
# Set up the legoESM Python environment ON a Cloud TPU VM, then override jax
# with the TPU build + libtpu.
#
# The repo is a uv WORKSPACE: the federated members (legoesm-core / -atmosphere
# / -ocean / -land / -ice / -coupler / -ml / -tools) live in-tree under
# packages/* and resolve via [tool.uv.sources] (workspace = true).  A bare
# `pip install -e .` therefore tries to fetch e.g. legoesm-atmosphere~=0.1.0
# from PyPI (where it is not published) and fails.  We install with uv, which:
#   (a) installs every workspace member editable from uv.lock (reproducible),
#   (b) provisions a managed CPython >=3.11 when the VM's default python3 is too
#       old (Ubuntu 22.04 ships 3.10) -- so no deadsnakes/system Python needed.
#
# Run from the repo root on the TPU VM (after git clone):
#   cd ~/legoESM && bash scripts/cluster/gcp_tpu/setup_env.sh
#
# Knobs (env vars):
#   JAX_VERSION  TPU jax/jaxlib pin (default 0.10.1, the locally validated one;
#                uv.lock pins the CPU build at 0.10.0 -- we override to the TPU
#                build here).  Keep jax and the TPU jaxlib at the SAME version.
#   PY_VERSION   Python to provision for the venv (default 3.11; >=3.11 required).
#   VENV_DIR     venv location (default .venv).
set -euo pipefail

JAX_VERSION="${JAX_VERSION:-0.10.1}"
PY_VERSION="${PY_VERSION:-3.11}"
VENV_DIR="${VENV_DIR:-.venv}"

# Resolve repo root from this script's location so it works regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
echo "Repo root: $REPO_ROOT"

# 1. Ensure uv is available (the workspace installer that resolves the federated
#    members and provisions Python).
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv in ~/.local/bin; make it visible for this run.
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH after install. Add ~/.local/bin to PATH" \
       "(e.g. 'export PATH=\$HOME/.local/bin:\$PATH') and re-run." >&2
  exit 1
fi
echo "Using uv: $(uv --version)"

# 2. Sync the workspace: create $VENV_DIR, install every member editable plus
#    the locked deps, provisioning CPython $PY_VERSION if the system lacks it.
#    uv.lock is current, so this installs from the lock (reproducible).  If uv
#    complains the lock is out of date, re-run without UV-frozen semantics by
#    deleting uv.lock's drift -- but normally this just works.
echo "Syncing workspace (uv sync, python $PY_VERSION) ..."
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
uv sync --python "$PY_VERSION"

# 3. Override jax/jaxlib with the TPU build + libtpu.  MUST come after the sync
#    so the TPU jaxlib wins over the CPU one from the lock.  Use `uv pip`
#    (NOT `uv run`/`uv sync`, which would re-pin jax back to the locked CPU
#    build and silently undo this).
echo "Installing jax[tpu]==$JAX_VERSION ..."
uv pip install -U "jax[tpu]==${JAX_VERSION}" \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 4. Verify the TPU backend via the venv directly.  Do NOT use `uv run` here:
#    it would re-sync the env to the lock and undo the jax[tpu] override above.
echo
echo "Verifying TPU is visible to JAX ..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
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
echo "  N_GPUS=1 bash scripts/cluster/gcp_tpu/run_smoke.sh   # single-chip VM"
echo "  bash scripts/cluster/gcp_tpu/run_bench.sh            # 8-chip slice"