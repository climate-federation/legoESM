#!/usr/bin/env bash
# Operator-only kt=2 TKE acquisition. The default is read-only preflight.
set -Eeuo pipefail

readonly MODE=${1:---preflight-only}
case "$MODE" in
  --preflight-only|--run|--admit-existing) ;;
  *) printf 'REFUSE: usage: %s [--preflight-only|--run|--admit-existing]\n' "$0" >&2; exit 64 ;;
esac

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
readonly here
readonly REPO=$(CDPATH= cd -- "$here/../../../../.." && pwd -P)
readonly PARENT=$here/../nemo_testcase_l4_orca2_round328_dynzdf_acquisition/run.sh
readonly GATE=$here/../nemo_testcase_l4_orca2_round336_tke_gate.py
readonly STUB_RECORD_BUILDER=$here/make_stub_records.py
readonly TEST=$REPO/tests/ocean/fidelity/test_orca2_round336_tke_gate.py
readonly EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/acquisition
readonly TARGET_RUN=$EVIDENCE/orca1ice_tke_internal_kt2_np2
readonly TARGET_CFG=ORCA2_ORCA1ICE_OMIP_L4_R336TKE
readonly SOURCE_OUTPUT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2/ocean.output
readonly PY=/home/dbalwada/legoESM/.venv/bin/python

refuse() { printf 'REFUSE: %s\n' "$*" >&2; exit 65; }
plant_must_refuse() {
  local label=$1; shift
  if "$@"; then refuse "preflight plant stayed green: $label"; fi
  printf 'PREFLIGHT_PLANT_FIRED %s\n' "$label"
}

for path in "$PARENT" "$GATE" "$STUB_RECORD_BUILDER" "$TEST" "$SOURCE_OUTPUT"; do
  [[ -f $path ]] || refuse "missing round-336 TKE acquisition input $path"
done

scratch=$(mktemp -d /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round336/.r336-tke-driver.XXXXXX)
retention_armed=0
cleanup() {
  local rc=$? refused
  trap - EXIT
  set +e
  if [[ $rc -ne 0 && $retention_armed -ne 0 && -d $TARGET_RUN ]]; then
    refused=$EVIDENCE/refused/$(basename "$TARGET_RUN")_wrapper_refused_$(date -u +%Y%m%dT%H%M%SZ)_$$
    mkdir -p "$EVIDENCE/refused"
    if mv -- "$TARGET_RUN" "$refused"; then
      printf 'R336_TKE_FAILED_OUTPUTS_KEPT path=%s\n' "$refused" >&2
    else
      printf 'R336_TKE_FAILED_OUTPUTS_LEFT_IN_PLACE path=%s\n' "$TARGET_RUN" >&2
    fi
  fi
  [[ ! -d $scratch ]] || rm -rf -- "$scratch"
  exit "$rc"
}
trap cleanup EXIT

check_driver_body() {
  local driver=$1
  [[ -s $driver ]] || return 1
  grep -F 'case "$MODE" in' "$driver" >/dev/null || return 1
  grep -F 'cleanup() {' "$driver" >/dev/null || return 1
  grep -F 'r336_tke_stub_prepare_target "$TARGET_ROOT"' "$driver" >/dev/null || return 1
  grep -F 'r336_tke_stub_finish_target "$TARGET_ROOT" "$dry_bld"' "$driver" >/dev/null || return 1
  grep -F 'mpirun -np 2 --oversubscribe ./nemo' "$driver" >/dev/null || return 1
  grep -F 'run_completed=1' "$driver" >/dev/null || return 1
  grep -F 'admit' "$driver" >/dev/null || return 1
}

exercise_stub_flow() {
  local base=$scratch/flow stub=$scratch/stub-nemo stub_bin=$scratch/bin \
    records=$scratch/stub-records retained canonical
  "$PY" "$STUB_RECORD_BUILDER" --output "$records" \
    --tracer-reference-dir \
      /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_tracer_stage3_kt2_np2 \
    --ldf-reference-dir \
      /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round333/acquisition/orca1ice_traldf_iso_internal_kt2_np2 \
    --ice-reference-dir \
      /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
    --drag-reference-dir \
      /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round328/acquisition/orca1ice_dynzdf_kt1_np2
  mkdir -p "$base/success" "$base/refusal" "$base/fabricated" \
    "$base/retention" "$stub_bin"
  cat >"$stub" <<'STUB'
#!/usr/bin/env bash
set -eu
for name in ORCA2_00000010_restart_0000.nc ORCA2_00000010_restart_0001.nc \
  ORCA2_00000010_restart_ice_0000.nc ORCA2_00000010_restart_ice_0001.nc; do
  cp -- "$R336_TKE_STUB_DETERMINISTIC_SOURCE/$name" .
done
if [[ $(basename "$PWD") == observer_free_twin ]]; then
  cp -- "$R336_TKE_STUB_SOURCE_OUTPUT" ocean.output
  printf 'STOP 0\n'
  exit 0
fi
while IFS= read -r name; do
  cp -- "$R336_TKE_STUB_RECORD_ROOT/$name" .
done < <(sed -n 's/.*"\(oracle_[^"]*\.bin\)".*/\1/p' round336_output_contract.json)
if [[ -n ${R336_TKE_STUB_FABRICATED_OUTPUT:-} ]]; then
  printf 'fabricated\n' >"$R336_TKE_STUB_FABRICATED_OUTPUT"
fi
case "${R336_TKE_STUB_SCENARIO:-success}" in
  success)
    cp -- "$R336_TKE_STUB_SOURCE_OUTPUT" ocean.output
    printf 'success\n' >.r336-tke-stub-scenario
    printf 'STOP 0\n'
    ;;
  refusal)
    printf 'before\nE R R O R stub refusal\nafter\n' >ocean.output
    printf 'refusal\n' >.r336-tke-stub-scenario
    printf 'STOP 0\n'
    ;;
  retention)
    cp -- "$R336_TKE_STUB_SOURCE_OUTPUT" ocean.output
    printf 'retention\n' >.r336-tke-stub-scenario
    printf 'stub crash\n'
    exit 17
    ;;
  *) exit 64 ;;
esac
STUB
  chmod 700 "$stub"
  cat >"$stub_bin/mpirun" <<'STUB'
#!/usr/bin/env bash
set -eu
while [[ $# -gt 0 ]]; do
  if [[ $1 == ./nemo ]]; then exec "$@"; fi
  shift
done
exit 64
STUB
  chmod 700 "$stub_bin/mpirun"

  if ! R336_TKE_STUB_FLOW_ROOT=$base/success R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_SCENARIO=success R328_ACQ_TARGET=tke336 \
      "$PARENT" --run >"$base/success.log" 2>&1; then
    tail -80 "$base/success.log" >&2
    refuse 'stub success flow refused'
  fi
  grep -F 'R336_TKE_STUB_ADMISSION_PASS scenario=success' "$base/success.log" >/dev/null \
    || refuse 'stub success missed real admission branch'
  if ! R336_TKE_STUB_FLOW_ROOT=$base/success R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_SCENARIO=success R328_ACQ_TARGET=tke336 \
      "$PARENT" --admit-existing >"$base/in-place.log" 2>&1; then
    tail -120 "$base/in-place.log" >&2
    refuse 'stub in-place admission refused'
  fi

  if R336_TKE_STUB_FLOW_ROOT=$base/refusal R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_SCENARIO=refusal R328_ACQ_TARGET=tke336 \
      "$PARENT" --run >"$base/refusal.log" 2>&1; then
    refuse 'stub NEMO error flow stayed green'
  fi
  grep -F 'NEMO_ERROR_LINES_BEGIN' "$base/refusal.log" >/dev/null \
    || refuse 'stub NEMO error was not quoted'
  find "$base/refusal/evidence/refused" -mindepth 1 -maxdepth 1 -type d \
    -name 'orca1ice_tke_internal_kt2_np2_*' -print -quit | grep . >/dev/null \
    || refuse 'stub refusal outputs were not retained'

  if R336_TKE_STUB_FLOW_ROOT=$base/fabricated R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_FABRICATED_OUTPUT=oracle_impossible.bin \
      R336_TKE_STUB_SCENARIO=success R328_ACQ_TARGET=tke336 \
      "$PARENT" --run >"$base/fabricated.log" 2>&1; then
    refuse 'stub impossible-output plant stayed green'
  fi
  grep -F "unexpected=['oracle_impossible.bin']" "$base/fabricated.log" >/dev/null \
    || refuse 'stub impossible-output plant missed the output-set refusal'
  printf 'PREFLIGHT_PLANT_FIRED r336-tke-impossible-stub-output\n'

  if R336_TKE_STUB_FLOW_ROOT=$base/retention R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_SCENARIO=retention R328_ACQ_TARGET=tke336 \
      "$PARENT" --run >"$base/retention.log" 2>&1; then
    refuse 'crashing stub NEMO stayed green'
  fi
  retained=$(find "$base/retention/evidence/refused" -mindepth 1 -maxdepth 1 \
    -type d -name 'orca1ice_tke_internal_kt2_np2_*' -print -quit)
  [[ -n $retained ]] || refuse 'crashed stub outputs were not retained'
  if ! R336_TKE_STUB_FLOW_ROOT=$base/retention R336_TKE_STUB_NEMO=$stub \
      R336_TKE_STUB_BIN=$stub_bin R336_TKE_STUB_SOURCE_OUTPUT=$SOURCE_OUTPUT \
      R336_TKE_STUB_RECORD_ROOT=$records \
      R336_TKE_STUB_DETERMINISTIC_SOURCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round326/acquisition/orca1ice_tke_hmi_10step_np2 \
      R336_TKE_STUB_SCENARIO=success R328_ACQ_TARGET=tke336 \
      R328_EXISTING_RUN=$retained "$PARENT" --admit-existing \
      >"$base/retained-admit.log" 2>&1; then
    tail -80 "$base/retained-admit.log" >&2
    refuse 'retained stub admission refused'
  fi
  grep -F 'R336_TKE_STUB_ADMISSION_PASS scenario=retention' \
    "$base/retained-admit.log" >/dev/null || refuse 'retained admission missed the stub gate'
  canonical=$base/retention/evidence/orca1ice_tke_internal_kt2_np2
  [[ -d $canonical && ! -e $retained ]] \
    || refuse 'retained stub output was not moved into canonical place'
  printf 'R336_TKE_STUB_FLOW_PASS cases=success,in-place,refusal,impossible-output,retention,retained-admission\n'
}

export R328_ACQ_TARGET=tke336

if [[ $MODE == --preflight-only ]]; then
  bash -n "$0" || refuse 'round-336 wrapper has invalid shell syntax'
  bash -n "$PARENT" || refuse 'round-336 parent has invalid shell syntax'
  cp -- "$PARENT" "$scratch/parent.syntax-plant.sh"
  printf '\nif then\n' >>"$scratch/parent.syntax-plant.sh"
  plant_must_refuse r336-tke-parent-syntax bash -n "$scratch/parent.syntax-plant.sh"
  check_driver_body "$PARENT" || refuse 'round-336 parent driver body is incomplete'
  awk '/^admit\(\) \{/ {exit} {print}' "$PARENT" >"$scratch/parent.body-truncated.sh"
  bash -n "$scratch/parent.body-truncated.sh" || refuse 'body truncation plant is not valid shell'
  plant_must_refuse r336-tke-parent-body check_driver_body "$scratch/parent.body-truncated.sh"
  export PYTHONPATH=$REPO:$REPO/packages/core:$REPO/packages/ocean:$REPO/packages/atmosphere:$REPO/packages/coupler:$REPO/packages/ice:$REPO/packages/land:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
  "$PY" -m pytest -q -p no:cacheprovider "$TEST"
  "$PARENT" --preflight-only
  exercise_stub_flow
  printf 'ORCA2_ROUND336_TKE_KT2_PREFLIGHT_READY %s\n' "$TARGET_RUN"
  exit 0
fi

retention_armed=1
"$PARENT" "$MODE"
retention_armed=0
printf 'ORCA2_ROUND336_TKE_KT2_ACQUISITION_PASS %s\n' "$TARGET_RUN"
