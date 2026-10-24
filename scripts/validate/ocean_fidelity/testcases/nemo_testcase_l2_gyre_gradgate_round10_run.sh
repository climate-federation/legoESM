#!/usr/bin/env bash
set -Eeuo pipefail

readonly HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --self-test ]]; then
  exec "$HERE/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh" --self-test
fi
readonly EVIDENCE_DIR=${1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round10/operator_rows}
exec "$HERE/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh" "$EVIDENCE_DIR"
