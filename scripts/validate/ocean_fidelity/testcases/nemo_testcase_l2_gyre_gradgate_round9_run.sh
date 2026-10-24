#!/usr/bin/env bash
set -Eeuo pipefail

readonly HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$HERE/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh" "$@"
