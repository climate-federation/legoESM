#!/usr/bin/env bash
# ORCA2 Round 342 — Standalone ORCA2 One-Year (2,920-step, S4 statistical arm)
# Observer-free K0 NEMO Twin Acquisition Runner
#
# Modes:
#   --preflight-only                  Default. Runs all Checklist 1-16 preflight
#                                     gates, Fortran fixture, FP census, and
#                                     end-to-end stub pipeline without running NEMO.
#   --run                             Operator-only. Builds observer-free K0 config
#                                     ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y (empty MY_SRC),
#                                     runs reproducibility Twin A (orca2_k0_year1_a_np2)
#                                     and candidate Twin B (orca2_k0_year1_b_np2) for
#                                     2,920 steps (nn_stock=8 daily restarts, 5,400 s
#                                     timeout per arm), and admits both twins.
#   --admit-existing <run_root>       Admits existing Twin A + Twin B in-place, or
#                                     quarantines failed runs to <run_root>/refused/.
#   --stub-run-dir <stub_root>        Replaces ONLY the ./nemo binary inside the real
#                                     staging + mpirun + admission pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
GATE_PY="${REPO_ROOT}/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_gate.py"
PYTHON_BIN="/home/dbalwada/legoESM/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export PATH="/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/packages/ocean:${REPO_ROOT}/packages/core:${REPO_ROOT}:${PYTHONPATH:-}"
export JAX_PLATFORMS="cpu"
export CUDA_VISIBLE_DEVICES=""
export JAX_ENABLE_X64=1

NEMO_ROOT="/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2"
TARGET_CFG_NAME="ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y"
BUILD_CFG="${NEMO_ROOT}/cfgs/${TARGET_CFG_NAME}"
RUN_ROOT="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year"
TWIN_A_NAME="orca2_k0_year1_a_np2"
TWIN_B_NAME="orca2_k0_year1_b_np2"
ICE_COVER_PATH="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
EVIDENCE_DIR="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342"
PER_ARM_TIMEOUT=5400
EXPECTED_STEPS=2920

MODE="preflight"
ADMIT_ROOT=""
STUB_ROOT=""
PLANT_STUB_UNPRODUCED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only)
      MODE="preflight"
      shift
      ;;
    --run)
      MODE="run"
      shift
      ;;
    --admit-existing)
      MODE="admit_existing"
      ADMIT_ROOT="$2"
      shift 2
      ;;
    --stub-run-dir)
      MODE="stub"
      STUB_ROOT="$2"
      shift 2
      ;;
    --stub-steps)
      EXPECTED_STEPS="$2"
      shift 2
      ;;
    --plant-stub-unproduced-file)
      PLANT_STUB_UNPRODUCED=1
      shift
      ;;
    --ice-cover)
      ICE_COVER_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

check_memory_and_processes() {
  local avail_gb
  avail_gb="$(free -g | awk '/^Mem:/ {print $7}')"
  if [[ -z "${avail_gb}" || "${avail_gb}" -lt 150 ]]; then
    echo "REFUSE: available RAM is ${avail_gb:-0} GiB (< 150 GiB required by Memory Rule)" >&2
    exit 1
  fi
  if pgrep -f '(^|/)nemo(\.exe)?([[:space:]]|$)' >/dev/null 2>&1; then
    echo "REFUSE: an active NEMO process is already running on this machine" >&2
    exit 1
  fi
}

build_observer_free_k0_nemo() {
  check_memory_and_processes
  local source_cfg="ORCA2_STANDALONE_IIF_OMIP_L4_R339K0"
  local source_root="${NEMO_ROOT}/cfgs/${source_cfg}"
  echo "Building observer-free K0 config ${TARGET_CFG_NAME} in ${NEMO_ROOT} ..."
  (
    cd "${NEMO_ROOT}"
    ./makenemo -r ORCA2_ICE_PISCES -n "${TARGET_CFG_NAME}" -m conda-scalarmath del_key 'key_xios'
    while IFS= read -r -d '' source; do
      cp -a "${source}" "${BUILD_CFG}/EXP00/$(basename "${source}")"
    done < <(find "${source_root}/EXP00" -maxdepth 1 \( -type f -o -type l \) -print0 | sort -z)
    cp "${source_root}/cpp_${source_cfg}.fcm" "${BUILD_CFG}/cpp_${TARGET_CFG_NAME}.fcm"
    ./makenemo -n "${TARGET_CFG_NAME}" -m conda-scalarmath del_key 'key_xios'
  )
  # Observer-free K0 requirement: MY_SRC must be empty.
  if [[ -d "${BUILD_CFG}/MY_SRC" ]]; then
    local extra_src
    extra_src="$(find "${BUILD_CFG}/MY_SRC" -maxdepth 1 -type f | wc -l)"
    if [[ "${extra_src}" -ne 0 ]]; then
      echo "REFUSE: ${BUILD_CFG}/MY_SRC is non-empty (${extra_src} files)" >&2
      exit 1
    fi
  fi
  if [[ ! -x "${BUILD_CFG}/BLD/bin/nemo.exe" ]]; then
    echo "REFUSE: build did not produce ${BUILD_CFG}/BLD/bin/nemo.exe" >&2
    exit 1
  fi
}

stage_single_arm() {
  local arm_dir="$1"
  local nemo_bin="$2"
  rm -rf "${arm_dir}"
  mkdir -p "${arm_dir}"
  "${PYTHON_BIN}" "${GATE_PY}" \
    --stage-run-dir "${arm_dir}" \
    --ice-cover "${ICE_COVER_PATH}" \
    --expected-steps "${EXPECTED_STEPS}" >/dev/null
  if [[ "${nemo_bin}" == "__STUB__" ]]; then
    local extra_flag=""
    if [[ "${PLANT_STUB_UNPRODUCED}" -eq 1 ]]; then
      extra_flag="--plant-stub-unproduced-file"
    fi
    cat > "${arm_dir}/nemo" <<EOF
#!/usr/bin/env bash
exec "${PYTHON_BIN}" "${GATE_PY}" --act-as-nemo-binary ${extra_flag}
EOF
    chmod +x "${arm_dir}/nemo"
  else
    cp -a "${nemo_bin}" "${arm_dir}/nemo"
  fi
}

run_single_arm() {
  local arm_dir="$1"
  local timeout_s="$2"
  if ! (
    cd "${arm_dir}"
    timeout --signal=TERM --kill-after=30s "${timeout_s}" \
      mpirun -np 2 --oversubscribe ./nemo \
      >mpirun.stdout.log 2>mpirun.stderr.log
  ); then
    echo "REFUSE: mpirun failed in ${arm_dir}:" >&2
    cat "${arm_dir}/mpirun.stdout.log" "${arm_dir}/mpirun.stderr.log" >&2 || true
    return 1
  fi
}

run_twin_pipeline() {
  local target_root="$1"
  local nemo_bin="$2"
  local timeout_s="$3"
  local arm_a="${target_root}/${TWIN_A_NAME}"
  local arm_b="${target_root}/${TWIN_B_NAME}"
  mkdir -p "${target_root}"
  stage_single_arm "${arm_a}" "${nemo_bin}"
  stage_single_arm "${arm_b}" "${nemo_bin}"
  run_single_arm "${arm_a}" "${timeout_s}"
  run_single_arm "${arm_b}" "${timeout_s}"
  "${PYTHON_BIN}" "${GATE_PY}" \
    --admit-existing "${target_root}" \
    --build-cfg "${BUILD_CFG}" \
    --expected-steps "${EXPECTED_STEPS}"
}

case "${MODE}" in
  preflight)
    "${PYTHON_BIN}" "${GATE_PY}" \
      --preflight-only \
      --evidence-dir "${EVIDENCE_DIR}" \
      --build-cfg "${BUILD_CFG}" \
      --ice-cover "${ICE_COVER_PATH}"
    echo "ORCA2_ROUND342_S4_YEAR_PREFLIGHT_READY"
    ;;
  stub)
    if [[ -z "${STUB_ROOT}" ]]; then
      echo "REFUSE: --stub-run-dir requires a path" >&2
      exit 2
    fi
    run_twin_pipeline "${STUB_ROOT}" "__STUB__" 300
    ;;
  admit_existing)
    if [[ -z "${ADMIT_ROOT}" ]]; then
      echo "REFUSE: --admit-existing requires a path" >&2
      exit 2
    fi
    "${PYTHON_BIN}" "${GATE_PY}" \
      --admit-existing "${ADMIT_ROOT}" \
      --build-cfg "${BUILD_CFG}" \
      --expected-steps "${EXPECTED_STEPS}"
    ;;
  run)
    check_memory_and_processes
    "${PYTHON_BIN}" "${GATE_PY}" \
      --preflight-only \
      --evidence-dir "${EVIDENCE_DIR}" \
      --build-cfg "${BUILD_CFG}" \
      --ice-cover "${ICE_COVER_PATH}"
    build_observer_free_k0_nemo
    run_twin_pipeline "${RUN_ROOT}" "${BUILD_CFG}/BLD/bin/nemo.exe" "${PER_ARM_TIMEOUT}"
    echo "ORCA2_ROUND342_S4_YEAR_ACQUISITION_COMPLETE: ${RUN_ROOT}"
    ;;
esac
