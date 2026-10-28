#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT=$(cd "$(dirname "$0")/../../../../.." && pwd)
readonly PY=/home/dbalwada/legoESM/.venv/bin/python
readonly BENCH=$REPO_ROOT/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round343_speed_benchmark.py
readonly OUTPUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r344
readonly OLD_REPORT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343/orca2_round343_speed_benchmark_report.json
readonly PYTHONPATH_VALUE=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src

refuse() {
  printf 'REFUSE_R344_GPU %s\n' "$*" >&2
  exit 64
}

[[ ${1:-} == --run && $# == 2 ]] || {
  printf 'usage: %s --run GPU_INDEX\n' "$0" >&2
  exit 2
}
readonly GPU_INDEX=$2
[[ $GPU_INDEX =~ ^[0-9]+$ ]] || refuse "GPU_INDEX must be a non-negative integer"
[[ -x $PY ]] || refuse "missing Python environment: $PY"
[[ -f $BENCH ]] || refuse "missing reused round-343 benchmark: $BENCH"
[[ -f $OUTPUT/arm_fast_literal_cpu_fp64.json ]] || refuse "CPU arm is absent"
[[ -f $OLD_REPORT ]] || refuse "round-343 baseline report is absent"

query=$(nvidia-smi \
  --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits) || refuse "nvidia-smi cannot query the driver"
row=$(awk -F, -v target="$GPU_INDEX" '
  { gsub(/[[:space:]]/, "", $1); if ($1 == target) { print; found=1 } }
  END { if (!found) exit 1 }
' <<<"$query") || refuse "GPU index $GPU_INDEX is unavailable"
IFS=, read -r found_index memory_used utilization <<<"$row"
memory_used=${memory_used//[[:space:]]/}
utilization=${utilization//[[:space:]]/}
[[ $memory_used == 0 && $utilization == 0 ]] || refuse \
  "GPU $GPU_INDEX is busy: memory.used=${memory_used}MiB utilization=${utilization}%"

cd "$REPO_ROOT"
export PYTHONPATH=$PYTHONPATH_VALUE
export JAX_ENABLE_X64=1
export CUDA_VISIBLE_DEVICES=$GPU_INDEX
unset JAX_PLATFORMS

"$PY" "$BENCH" \
  --arm fast_literal_gpu_fp64 \
  --output-dir "$OUTPUT"
"$PY" "$BENCH" \
  --arm assemble-fast \
  --output-dir "$OUTPUT" \
  --round343-report "$OLD_REPORT"

printf 'PASS_R344_FAST_LITERAL_GPU gpu=%s report=%s\n' \
  "$GPU_INDEX" "$OUTPUT/orca2_round344_fast_literal_benchmark_report.json"
