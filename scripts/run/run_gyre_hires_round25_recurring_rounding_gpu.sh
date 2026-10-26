#!/usr/bin/env bash
set -Eeuo pipefail

[[ ${1:-} == --physical-gpu && ${2:-} =~ ^[0-9]+$ \
   && ${3:-} == --seed-root && -n ${4:-} \
   && ${5:-} == --output && -n ${6:-} ]] || {
  printf 'REFUSE_GPU: usage: %s --physical-gpu PHYSICAL_INDEX --seed-root ABSOLUTE_PATH --output ABSOLUTE_PATH\n' "$0" >&2
  exit 64
}
physical_gpu=$2
seed_root=$4
output=$6
[[ $seed_root == /* && $output == /* ]] || {
  printf 'REFUSE_PATH: --seed-root and --output must be absolute\n' >&2
  exit 64
}
[[ -f $seed_root/seed_manifest.json ]] || {
  printf 'REFUSE_SEED: seed manifest is absent: %s\n' "$seed_root" >&2
  exit 65
}
[[ ! -e $output ]] || {
  printf 'REFUSE_OUTPUT: output already exists: %s\n' "$output" >&2
  exit 65
}

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact round-25 instrument first\n' >&2
  exit 66
}
mkdir -p "$(dirname -- "$output")"
export JAX_ENABLE_X64=1
export PYTHONPATH=$repo_root/packages/core:$repo_root/packages/ocean:$repo_root/packages/atmosphere:$repo_root/packages/coupler:$repo_root/packages/ice:$repo_root/packages/land:$repo_root/packages/ml:$repo_root/packages/tools:$repo_root/src
instrument=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_round24_rounding.py

CUDA_VISIBLE_DEVICES=$physical_gpu \
  /home/dbalwada/legoESM/.venv/bin/python "$instrument" run-recurring \
  --seed-root "$seed_root" --output "$output/run" \
  --physical-gpu "$physical_gpu"

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  /home/dbalwada/legoESM/.venv/bin/python "$instrument" score \
  --run-root "$output/run" --output "$output/score"

printf 'ACQUISITION_DONE: %s physical_gpu=%s days=180 arms=2 seed_applications=4320\n' \
  "$output" "$physical_gpu"
