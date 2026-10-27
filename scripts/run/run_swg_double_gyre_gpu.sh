#!/usr/bin/env bash
set -Eeuo pipefail

export JAX_ENABLE_X64=1
unset JAX_PLATFORMS

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
python=${PYTHON:-}
[[ -n "$python" ]] || {
  printf 'REFUSE_PYTHON: set PYTHON to the pinned environment interpreter\n' >&2
  exit 64
}
source "$repo_root/scripts/data/nemo_swg_double_gyre/checkout_env.bash"
swg_dg_bind_checkout "$python"
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact GPU driver first\n' >&2
  exit 66
}
export PYTHONPATH=$repo_root/packages/core:$repo_root/packages/ocean:$repo_root/packages/atmosphere:$repo_root/packages/coupler:$repo_root/packages/ice:$repo_root/packages/land:$repo_root/packages/ml:$repo_root/packages/tools:$repo_root/src
exec "$python" \
  "$here/run_swg_double_gyre.py" --mode gpu-native "$@"
