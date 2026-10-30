#!/usr/bin/env bash
# ORCA2 Round 342 — Detached CPU launcher for the standalone K0 legoESM arm.
#
# Usage:
#   # 20-step smoke (detached on CPU):
#   bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh \
#     --steps 20 --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_smoke_20step
#
#   # Full 2,920-step (1-year) run (operator-gated via --allow-full-year):
#   bash scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_legoesm_run.sh \
#     --steps 2920 --allow-full-year \
#     --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_year1_2920step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DRIVER_PY="${REPO_ROOT}/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_driver.py"
PYTHON_BIN="/home/dbalwada/legoESM/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

STEPS=20
OUTPUT_DIR="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_smoke_20step"
ALLOW_FULL_YEAR=0
WAIT_FOR_COMPLETION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --allow-full-year)
      ALLOW_FULL_YEAR=1
      shift
      ;;
    --wait)
      WAIT_FOR_COMPLETION=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "${STEPS}" -gt 40 && "${ALLOW_FULL_YEAR}" -ne 1 ]]; then
  echo "REFUSE: --steps ${STEPS} (> 40) requires explicit operator flag --allow-full-year" >&2
  exit 1
fi

AVAIL_GB="$(free -g | awk '/^Mem:/ {print $7}')"
if [[ -z "${AVAIL_GB}" || "${AVAIL_GB}" -lt 150 ]]; then
  echo "REFUSE: available RAM is ${AVAIL_GB:-0} GiB (< 150 GiB required by Memory Rule)" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/legoesm_run.log"
PID_FILE="${OUTPUT_DIR}/legoesm_run.pid"

export PYTHONPATH="${REPO_ROOT}/packages/ocean:${REPO_ROOT}/packages/core:${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

nohup "${PYTHON_BIN}" "${DRIVER_PY}" \
  --steps "${STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  >"${LOG_FILE}" 2>&1 &
BG_PID=$!
echo "${BG_PID}" > "${PID_FILE}"

echo "LAUNCHED_DETACHED_LEGOESM_K0: pid=${BG_PID} backend=cpu steps=${STEPS} output_dir=${OUTPUT_DIR}"

if [[ "${WAIT_FOR_COMPLETION}" -eq 1 ]]; then
  rc=0
  wait "${BG_PID}" || rc=$?
  cat "${LOG_FILE}"
  exit "${rc}"
fi
