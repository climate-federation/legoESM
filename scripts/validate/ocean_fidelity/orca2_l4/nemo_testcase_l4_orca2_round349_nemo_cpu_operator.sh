#!/usr/bin/env bash
# ORCA2 round 349 — operator-only NEMO 5.0.2 per-core CPU sweep.
#
# This script never builds NEMO. It copies the certified executable/config,
# symlinks its inputs, disables diagnostic/restart writes, runs 96 steps, and
# parses NEMO's own steady timing.output block.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BENCHMARK_PY="${SCRIPT_DIR}/nemo_testcase_l4_orca2_round343_speed_benchmark.py"
RANK_WRAPPER="${SCRIPT_DIR}/nemo_testcase_l4_orca2_round349_rank_wrapper.sh"
PYTHON_BIN="/home/dbalwada/legoESM/.venv/bin/python"
SCRATCH_ROOT="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r349_scratch"
SOURCE_RUN="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2"
OUTPUT_ROOT="${SCRATCH_ROOT}/nemo_cpu_sweep"
CORE_COUNTS=(1 2 4 8 16 40)
# Guards against initial-state diagnostic output during the timed short run.
NEMO_NN_ISTATE=0
# Guards against periodic/final restart output during the timed short run.
NEMO_NN_STOCK=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "REFUSE: unknown argument $1" >&2
      exit 2
      ;;
  esac
done

export TMPDIR="${SCRATCH_ROOT}"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64=1
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/packages/core:${REPO_ROOT}/packages/ocean:${REPO_ROOT}/packages/atmosphere:${REPO_ROOT}/packages/coupler:${REPO_ROOT}/packages/ice:${REPO_ROOT}/packages/land:${REPO_ROOT}/packages/ml:${REPO_ROOT}/packages/tools:${REPO_ROOT}/src"

case "$(realpath -m "${OUTPUT_ROOT}")" in
  "${SCRATCH_ROOT}"/*) ;;
  *)
    echo "REFUSE: output root must be below ${SCRATCH_ROOT}: ${OUTPUT_ROOT}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "REFUSE: required Python is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${SOURCE_RUN}/nemo" ]]; then
  echo "REFUSE: certified NEMO executable is absent: ${SOURCE_RUN}/nemo" >&2
  exit 1
fi
if [[ ! -f "${SOURCE_RUN}/producer_manifest.json" ]]; then
  echo "REFUSE: certified NEMO producer manifest is absent" >&2
  exit 1
fi
EXPECTED_BINARY_SHA="$(timeout 1200 "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["binary_sha256"])' "${SOURCE_RUN}/producer_manifest.json")"
ACTUAL_BINARY_SHA="$(sha256sum "${SOURCE_RUN}/nemo" | awk '{print $1}')"
if [[ "${ACTUAL_BINARY_SHA}" != "${EXPECTED_BINARY_SHA}" ]]; then
  echo "REFUSE: certified NEMO binary digest disagrees with producer manifest" >&2
  exit 1
fi

MPIRUN_BIN="${MPIRUN_BIN:-mpirun}"
GNU_TIME_BIN="${GNU_TIME_BIN:-/usr/bin/time}"
if [[ "${DRY_RUN}" -eq 0 ]]; then
  if ! command -v "${MPIRUN_BIN}" >/dev/null 2>&1; then
    echo "REFUSE: Open MPI launcher not found: ${MPIRUN_BIN}" >&2
    exit 1
  fi
  if ! "${MPIRUN_BIN}" --version 2>&1 | grep -qi "Open MPI"; then
    echo "REFUSE: round 349 requires Open MPI --cpu-set/--bind-to semantics" >&2
    exit 1
  fi
  if [[ ! -x "${GNU_TIME_BIN}" ]]; then
    echo "REFUSE: GNU time is absent: ${GNU_TIME_BIN}" >&2
    exit 1
  fi
  if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "REFUSE: output root must be absent or empty: ${OUTPUT_ROOT}" >&2
    exit 1
  fi
  mkdir -p "${OUTPUT_ROOT}"
fi

for CORES in "${CORE_COUNTS[@]}"; do
  CPU_IDS="$(timeout 1200 "${PYTHON_BIN}" "${BENCHMARK_PY}" \
    --round349-print-cpu-ids --round349-cores "${CORES}")"
  RUN_DIR="${OUTPUT_ROOT}/nemo_n${CORES}"
  PREFLIGHT="${OUTPUT_ROOT}/preflight_n${CORES}.json"
  ARTIFACT="${OUTPUT_ROOT}/nemo_n${CORES}.json"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "ROUND349_NEMO_DRY_RUN n=${CORES} cpu_ids=${CPU_IDS} run_dir=${RUN_DIR}"
    echo "  preflight: RAM>=150GiB; no nemo; selected-core utilization<=10%/10s; no unrelated process>=1 core"
    echo "  config: nn_itend=96 nn_istate=${NEMO_NN_ISTATE} nn_stock=${NEMO_NN_STOCK} nn_write=0"
    echo "  command: (cd ${RUN_DIR} && ROUND349_CPU_IDS=${CPU_IDS} ${GNU_TIME_BIN} -o gnu_time.txt ${MPIRUN_BIN} --report-bindings --bind-to none -np ${CORES} ./round349_rank_wrapper.sh)"
    continue
  fi

  timeout 1200 "${PYTHON_BIN}" "${BENCHMARK_PY}" \
    --round349-preflight \
    --round349-cpu-ids "${CPU_IDS}" \
    --round349-sample-seconds 10 \
    --round349-output-file "${PREFLIGHT}" >/dev/null

  mkdir "${RUN_DIR}"
  cp -p "${SOURCE_RUN}/nemo" "${RUN_DIR}/nemo"
  cp -p "${RANK_WRAPPER}" "${RUN_DIR}/round349_rank_wrapper.sh"
  cp -p "${SOURCE_RUN}"/namelist* "${RUN_DIR}/"
  cp -p "${SOURCE_RUN}"/*.xml "${RUN_DIR}/"
  cp -p "${SOURCE_RUN}/compiled_keys.txt" "${SOURCE_RUN}/producer_manifest.json" "${RUN_DIR}/"
  while IFS= read -r INPUT_LINK; do
    ln -s "$(readlink "${INPUT_LINK}")" "${RUN_DIR}/$(basename "${INPUT_LINK}")"
  done < <(find "${SOURCE_RUN}" -maxdepth 1 -type l -print | sort)
  ln -s "${SOURCE_RUN}/ice_cover_clim.nc" "${RUN_DIR}/ice_cover_clim.nc"

  sed -E -i \
    -e 's/^([[:space:]]*nn_itend[[:space:]]*=[[:space:]]*)[0-9]+/\1 96/' \
    -e "s/^([[:space:]]*nn_istate[[:space:]]*=[[:space:]]*)[0-9]+/\1 ${NEMO_NN_ISTATE}/" \
    -e "s/^([[:space:]]*nn_stock[[:space:]]*=[[:space:]]*)[0-9]+/\1 ${NEMO_NN_STOCK}/" \
    "${RUN_DIR}/namelist_cfg"
  grep -Eq '^[[:space:]]*nn_itend[[:space:]]*=[[:space:]]*96([[:space:]]|$)' "${RUN_DIR}/namelist_cfg"
  grep -Eq "^[[:space:]]*nn_istate[[:space:]]*=[[:space:]]*${NEMO_NN_ISTATE}([[:space:]]|$)" "${RUN_DIR}/namelist_cfg"
  grep -Eq "^[[:space:]]*nn_stock[[:space:]]*=[[:space:]]*${NEMO_NN_STOCK}([[:space:]]|$)" "${RUN_DIR}/namelist_cfg"

  (
    cd "${RUN_DIR}"
    ROUND349_CPU_IDS="${CPU_IDS}" timeout 1200 "${GNU_TIME_BIN}" \
      -f $'elapsed_s=%e\nuser_s=%U\nsystem_s=%S\nmax_rss_kib=%M' \
      -o gnu_time.txt \
      "${MPIRUN_BIN}" --report-bindings --bind-to none \
      -np "${CORES}" ./round349_rank_wrapper.sh \
      >mpirun.stdout.log 2>mpirun.stderr.log
  )

  timeout 1200 "${PYTHON_BIN}" "${BENCHMARK_PY}" \
    --round349-parse-nemo \
    --round349-nemo-run-dir "${RUN_DIR}" \
    --round349-preflight-path "${PREFLIGHT}" \
    --round349-cores "${CORES}" \
    --round349-cpu-ids "${CPU_IDS}" \
    --round349-output-file "${ARTIFACT}"
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "ROUND349_NEMO_DRY_RUN_COMPLETE arms=${#CORE_COUNTS[@]} source=${SOURCE_RUN}"
else
  echo "ROUND349_NEMO_SWEEP_COMPLETE arms=${#CORE_COUNTS[@]} output_root=${OUTPUT_ROOT}"
fi
