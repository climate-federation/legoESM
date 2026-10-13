#!/usr/bin/env bash
set -Eeuo pipefail

[[ $# -ge 1 && $1 =~ ^[0-9]+$ ]] || {
  printf 'USAGE: %s PHYSICAL_GPU [driver arguments]\n' "$0" >&2
  exit 64
}
physical_gpu=$1
shift
export CUDA_VISIBLE_DEVICES=$physical_gpu
export JAX_ENABLE_X64=1
unset JAX_PLATFORMS

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact GPU driver first\n' >&2
  exit 66
}
export PYTHONPATH=$repo_root/packages/core:$repo_root/packages/ocean:$repo_root/packages/atmosphere:$repo_root/packages/coupler:$repo_root/packages/ice:$repo_root/packages/land:$repo_root/packages/ml:$repo_root/packages/tools:$repo_root/src
exec /home/dbalwada/legoESM/.venv/bin/python \
  "$here/run_swg_double_gyre.py" --mode gpu-native \
  --physical-gpu "$physical_gpu" "$@"
