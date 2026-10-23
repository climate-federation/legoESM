#!/usr/bin/env bash
set -Eeuo pipefail

[[ ${1:-} == --physical-gpu && ${2:-} =~ ^[0-9]+$ ]] || {
  printf 'REFUSE_GPU: usage: %s --physical-gpu PHYSICAL_INDEX --output PATH [driver options]\n' "$0" >&2
  exit 64
}
physical_gpu=$2
shift 2
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
exec /home/dbalwada/legoESM/.venv/bin/python "$here/run_gyre_hires.py" \
  --deck r4_smag --days 1825 --mode gpu-native --physical-gpu "$physical_gpu" \
  --daily-means "$@"
