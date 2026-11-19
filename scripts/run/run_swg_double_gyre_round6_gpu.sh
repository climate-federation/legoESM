#!/usr/bin/env bash
set -Eeuo pipefail

[[ ${CUDA_VISIBLE_DEVICES:-1} == 1 ]] || {
  printf 'REFUSE_GPU: SWG round 6 is pinned to physical GPU 1\n' >&2
  exit 64
}
export CUDA_VISIBLE_DEVICES=1
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cuda

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(git -C "$here" rev-parse --show-toplevel)
[[ -z "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]] || {
  printf 'REFUSE_DIRTY_COMMIT: commit the exact GPU probe first\n' >&2
  exit 66
}
export PYTHONPATH=$repo_root/packages/core:$repo_root/packages/ocean:$repo_root/packages/atmosphere:$repo_root/packages/coupler:$repo_root/packages/ice:$repo_root/packages/land:$repo_root/packages/ml:$repo_root/packages/tools:$repo_root/src

evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swgdg_rounds/round6
probe=$repo_root/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_swg_dg_round6.py
python=/home/dbalwada/legoESM/.venv/bin/python

"$python" "$probe" replay --deck maddison --plant \
  --npz "$evidence/gpu_replay.npz" \
  --json "$evidence/gpu_replay.json"
"$python" "$probe" compare \
  "$evidence/cpu_default_replay.npz" "$evidence/gpu_replay.npz" \
  --json "$evidence/cpu_gpu_trace_comparison.json"
