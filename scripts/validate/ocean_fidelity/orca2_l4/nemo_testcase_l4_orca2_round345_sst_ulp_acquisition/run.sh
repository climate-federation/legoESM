#!/usr/bin/env bash
# ORCA2 round 345 — exact restart control plus +1-fp64-ULP SST continuation.
# Default: inherited round-342 plus round-345 preflight only.  --run is operator-only.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
GATE_PY="${REPO_ROOT}/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round345_sst_ulp_gate.py"
R342_GATE_PY="${REPO_ROOT}/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round342_s4_year_gate.py"
PYTHON_BIN="/home/dbalwada/legoESM/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python3"; fi

export PATH="/home/dbalwada/miniconda3/envs/nemo-build/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}/packages/ocean:${REPO_ROOT}/packages/core:${REPO_ROOT}:${PYTHONPATH:-}"
export JAX_PLATFORMS="cpu"
export CUDA_VISIBLE_DEVICES=""
export JAX_ENABLE_X64=1

BUILD_CFG="/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_STANDALONE_IIF_OMIP_L4_R342S4Y"
BASELINE_ARM="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round342/s4_year/orca2_k0_year1_a_np2"
RUN_ROOT="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round345/s4_year"
EVIDENCE_DIR="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r345"
ICE_COVER_PATH="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
CONTROL_ARM_NAME="orca2_k0_year1_restart_control_np2"
PERTURBED_ARM_NAME="orca2_k0_year1_sst_plus_1ulp_np2"
EXPECTED_STEPS=2920
PER_ARM_TIMEOUT=5400
REQUIRED_RAM_GIB="${R345_REQUIRED_RAM_GIB:-150}"
REQUIRED_DISK_GIB="${R345_REQUIRED_DISK_GIB:-25}"

MODE="preflight"
STUB_ROOT=""
ADMIT_ROOT=""
PLANT_STUB_UNPRODUCED=0
PLANT_STUB_MISSING_STOP=0
PLANT_STUB_SHORT_TIMESTEP=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only) MODE="preflight"; shift ;;
    --run) MODE="run"; shift ;;
    --stub-run-dir) MODE="stub"; STUB_ROOT="${2:-}"; shift 2 ;;
    --admit-existing) MODE="admit"; ADMIT_ROOT="${2:-}"; shift 2 ;;
    --stub-steps) EXPECTED_STEPS="$2"; shift 2 ;;
    --stub-timeout) PER_ARM_TIMEOUT="$2"; shift 2 ;;
    --plant-stub-unproduced-file) PLANT_STUB_UNPRODUCED=1; shift ;;
    --plant-stub-missing-stop) PLANT_STUB_MISSING_STOP=1; shift ;;
    --plant-stub-short-timestep) PLANT_STUB_SHORT_TIMESTEP="$2"; shift 2 ;;
    --ice-cover) ICE_COVER_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

check_clean_worktree() {
  local check_repo="${R345_CHECK_REPO_ROOT:-${REPO_ROOT}}"
  local dirty
  dirty="$(git -C "${check_repo}" status --porcelain --untracked-files=all)"
  if [[ -n "${dirty}" ]]; then
    echo "REFUSE: git worktree ${check_repo} is dirty:" >&2
    echo "${dirty}" >&2
    return 1
  fi
}

check_memory_and_disk() {
  local target="$1"
  local available_gib
  available_gib="$(free -g | awk '/^Mem:/ {print $7}')"
  if [[ -z "${available_gib}" || "${available_gib}" -lt "${REQUIRED_RAM_GIB}" ]]; then
    echo "REFUSE: available RAM ${available_gib:-0} GiB < ${REQUIRED_RAM_GIB} GiB" >&2
    return 1
  fi
  local probe="${target}"
  while [[ ! -d "${probe}" && "${probe}" != "/" ]]; do probe="$(dirname "${probe}")"; done
  local available_kib required_kib
  available_kib="$(df -Pk "${probe}" | awk 'NR==2 {print $4}')"
  required_kib="$(( REQUIRED_DISK_GIB * 1024 * 1024 ))"
  if [[ -z "${available_kib}" || "${available_kib}" -lt "${required_kib}" ]]; then
    echo "REFUSE: available disk at ${probe} is below ${REQUIRED_DISK_GIB} GiB" >&2
    return 1
  fi
}

check_no_active_nemo() {
  if pgrep -f '(^|/)nemo(\.exe)?([[:space:]]|$)' >/dev/null 2>&1; then
    echo "REFUSE: an active NEMO process already exists" >&2
    return 1
  fi
}

quarantine_arm() {
  local arm_dir="$1"
  local owner_token="$2"
  if [[ -d "${arm_dir}" ]] \
     && [[ -f "${arm_dir}/.round345_attempt" ]] \
     && [[ "$(<"${arm_dir}/.round345_attempt")" == "${owner_token}" ]]; then
    local refused timestamp destination
    refused="$(dirname "${arm_dir}")/refused"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${refused}"
    destination="${refused}/${timestamp}_$(basename "${arm_dir}").failed.$$"
    mv "${arm_dir}" "${destination}"
    echo "R345_REFUSED_ARM_QUARANTINED ${destination}" >&2
  fi
}

stage_arm() {
  local arm_dir="$1"
  local nemo_bin="$2"
  local arm_kind="$3"
  if [[ -e "${arm_dir}" || -L "${arm_dir}" ]]; then
    echo "REFUSE: retained arm already exists: ${arm_dir}" >&2
    return 1
  fi
  if ! mkdir "${arm_dir}"; then
    echo "REFUSE: could not claim new round-345 arm: ${arm_dir}" >&2
    return 1
  fi
  local owner_token="${R345_ATTEMPT_TOKEN:?missing R345_ATTEMPT_TOKEN}"
  printf '%s\n' "${owner_token}" >"${arm_dir}/.round345_attempt"
  local stage_flag="--stage-run-dir"
  if [[ "${arm_kind}" == "control" ]]; then stage_flag="--stage-control-run-dir"; fi
  "${PYTHON_BIN}" "${GATE_PY}" \
    "${stage_flag}" "${arm_dir}" \
    --build-cfg "${BUILD_CFG}" \
    --ice-cover "${ICE_COVER_PATH}" \
    --expected-steps "${EXPECTED_STEPS}" >/dev/null || return 1
  if [[ "${nemo_bin}" == "__STUB__" ]]; then
    local extra_flags=()
    if [[ "${PLANT_STUB_UNPRODUCED}" -eq 1 ]]; then extra_flags+=("--plant-stub-unproduced-file"); fi
    if [[ "${PLANT_STUB_MISSING_STOP}" -eq 1 ]]; then extra_flags+=("--plant-stub-missing-stop"); fi
    if [[ -n "${PLANT_STUB_SHORT_TIMESTEP}" ]]; then
      extra_flags+=("--plant-stub-short-timestep" "${PLANT_STUB_SHORT_TIMESTEP}")
    fi
    cat > "${arm_dir}/nemo" <<EOF
#!/usr/bin/env bash
exec "${PYTHON_BIN}" "${GATE_PY}" --act-as-nemo-binary ${extra_flags[*]:-}
EOF
    chmod +x "${arm_dir}/nemo"
  else
    cp -a "${nemo_bin}" "${arm_dir}/nemo"
  fi
  "${PYTHON_BIN}" "${GATE_PY}" --write-run-manifest "${arm_dir}" >/dev/null
}

run_arm() {
  local arm_dir="$1"
  local rc=0
  (
    cd "${arm_dir}"
    timeout --foreground --signal=TERM --kill-after=30s "${PER_ARM_TIMEOUT}s" \
      mpirun -np 2 --oversubscribe ./nemo >mpirun.stdout.log 2>mpirun.stderr.log
  ) || rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    echo "REFUSE: mpirun failed or timed out (rc=${rc}) in ${arm_dir}" >&2
    return "${rc}"
  fi
  if [[ ! -f "${arm_dir}/mpirun.stderr.log" ]] \
     || ! LC_ALL=C grep -Eq '(^|[[:space:]])STOP 0([[:space:]]|$)' "${arm_dir}/mpirun.stderr.log"; then
    echo "REFUSE: missing clean STOP 0 in ${arm_dir}/mpirun.stderr.log" >&2
    return 1
  fi
  local actual_step
  actual_step="$(tr -d '[:space:]' < "${arm_dir}/time.step")"
  if [[ "${actual_step}" != "${EXPECTED_STEPS}" ]]; then
    echo "REFUSE: short or mismatched time.step: ${actual_step} != ${EXPECTED_STEPS}" >&2
    return 1
  fi
  if LC_ALL=C grep -Eq 'E R R O R|STOP [1-9][0-9]*([^0-9]|$)' \
      "${arm_dir}/mpirun.stdout.log" "${arm_dir}/mpirun.stderr.log" "${arm_dir}/ocean.output"; then
    echo "REFUSE: runtime error marker in ${arm_dir}" >&2
    return 1
  fi
  "${PYTHON_BIN}" "${GATE_PY}" --write-run-manifest "${arm_dir}" >/dev/null
}

run_pipeline() {
  local root="$1"
  local binary="$2"
  local control_arm="${root}/${CONTROL_ARM_NAME}"
  local perturbed_arm="${root}/${PERTURBED_ARM_NAME}"
  mkdir -p "${root}"
  local existing
  for existing in "${control_arm}" "${perturbed_arm}"; do
    if [[ -e "${existing}" || -L "${existing}" ]]; then
      echo "REFUSE: retained arm already exists and will not be moved: ${existing}" >&2
      return 1
    fi
  done
  local owner_token="round345.$$.${RANDOM}"
  export R345_ATTEMPT_TOKEN="${owner_token}"
  local admission_flags=()
  if [[ "${binary}" == "__STUB__" ]]; then admission_flags+=("--allow-stub-binary"); fi
  if ! stage_arm "${control_arm}" "${binary}" "control" \
     || ! run_arm "${control_arm}" \
     || ! stage_arm "${perturbed_arm}" "${binary}" "perturbed" \
     || ! run_arm "${perturbed_arm}" \
     || ! "${PYTHON_BIN}" "${GATE_PY}" \
          --admit-existing "${root}" --build-cfg "${BUILD_CFG}" \
          --expected-steps "${EXPECTED_STEPS}" "${admission_flags[@]}"; then
    quarantine_arm "${control_arm}" "${owner_token}"
    quarantine_arm "${perturbed_arm}" "${owner_token}"
    return 1
  fi
  rm -f "${control_arm}/.round345_attempt" "${perturbed_arm}/.round345_attempt"
}

case "${MODE}" in
  preflight)
    check_clean_worktree
    check_memory_and_disk "${EVIDENCE_DIR}"
    "${PYTHON_BIN}" "${GATE_PY}" --preflight-only \
      --evidence-dir "${EVIDENCE_DIR}" --build-cfg "${BUILD_CFG}" \
      --ice-cover "${ICE_COVER_PATH}"
    echo "ORCA2_ROUND345_SST_ULP_PREFLIGHT_READY"
    ;;
  stub)
    [[ -n "${STUB_ROOT}" ]] || { echo "REFUSE: --stub-run-dir needs a path" >&2; exit 2; }
    check_clean_worktree
    check_memory_and_disk "${STUB_ROOT}"
    run_pipeline "${STUB_ROOT}" "__STUB__"
    ;;
  admit)
    [[ -n "${ADMIT_ROOT}" ]] || { echo "REFUSE: --admit-existing needs a path" >&2; exit 2; }
    check_clean_worktree
    admit_flags=()
    if [[ "${R345_PREFLIGHT_ALLOW_STUB:-0}" == "1" \
       && "${ADMIT_ROOT}" == /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/* ]]; then
      admit_flags+=("--allow-stub-binary")
    fi
    "${PYTHON_BIN}" "${GATE_PY}" --admit-existing "${ADMIT_ROOT}" \
      --build-cfg "${BUILD_CFG}" --expected-steps "${EXPECTED_STEPS}" "${admit_flags[@]}"
    ;;
  run)
    check_clean_worktree
    check_memory_and_disk "${RUN_ROOT}"
    check_no_active_nemo
    [[ -x "${BASELINE_ARM}/nemo" ]] || { echo "REFUSE: missing admitted round-342 binary" >&2; exit 1; }
    "${PYTHON_BIN}" "${GATE_PY}" --preflight-only \
      --evidence-dir "${EVIDENCE_DIR}" --build-cfg "${BUILD_CFG}" \
      --ice-cover "${ICE_COVER_PATH}"
    run_pipeline "${RUN_ROOT}" "${BASELINE_ARM}/nemo"
    echo "ORCA2_ROUND345_SST_ULP_ACQUISITION_COMPLETE: ${RUN_ROOT}/${CONTROL_ARM_NAME} + ${RUN_ROOT}/${PERTURBED_ARM_NAME}"
    ;;
esac
