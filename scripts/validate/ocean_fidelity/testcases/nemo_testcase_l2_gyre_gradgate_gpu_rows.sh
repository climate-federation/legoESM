#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "REFUSE ocean-gradgate-gpu: command failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2' ERR

readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly EVIDENCE_DIR=${1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round4}
readonly GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py
readonly MODEL_PATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src

refuse() {
  printf 'REFUSE ocean-gradgate-gpu: %s\n' "$1" >&2
  exit "${2:-64}"
}

accept_classified_exit() {
  local status=$1
  local rc=$2
  local label=$3
  case "$status" in
    CONFIRMED)
      [[ $rc -eq 0 ]] || refuse "CONFIRMED $label exited $rc" "$rc"
      ;;
    REFUTED|VACUOUS|RETRACTED)
      [[ $rc -eq 1 ]] || refuse "$status $label exited $rc" "$rc"
      ;;
    *)
      refuse "unknown $label status: $status" 66
      ;;
  esac
}

invoke_row_gate() {
  local output=$1
  shift
  if [[ -n ${SWG_GRADGATE_SELF_TEST_STATUS:-} ]]; then
    "$PYTHON" -c '
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": sys.argv[2],
    "measurement_policy_label": (
        "GPU, XLA transcendentals — not the certified bit-exact policy"),
    "precision_policy": "PrecisionPolicy.fp64(transcendentals=\x27native\x27)",
}))
' "$output" "$SWG_GRADGATE_SELF_TEST_STATUS"
    [[ $SWG_GRADGATE_SELF_TEST_STATUS == CONFIRMED ]]
    return
  fi
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
    PYTHONPATH="$MODEL_PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    env -u XLA_PYTHON_CLIENT_ALLOCATOR \
    "$PYTHON" "$GATE" "$@" --transcendentals native \
      --expect-commit "$COMMIT" --output "$output"
}

invoke_suite_gate() {
  local output=$1
  local steps=$2
  shift 2
  if [[ -n ${SWG_GRADGATE_SELF_TEST_STATUS:-} ]]; then
    "$PYTHON" -c '
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({"status": sys.argv[2]}))
' "$output" "$SWG_GRADGATE_SELF_TEST_STATUS"
    [[ $SWG_GRADGATE_SELF_TEST_STATUS == CONFIRMED ]]
    return
  fi
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
    PYTHONPATH="$MODEL_PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    env -u XLA_PYTHON_CLIENT_ALLOCATOR \
    "$PYTHON" "$GATE" resource-suite --card SWG --steps "$steps" \
      --expect-commit "$COMMIT" --output "$output" "$@"
}

run_row() {
  local output=$1
  shift
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  local rc=0
  if invoke_row_gate "$output" "$@"; then
    rc=0
  else
    rc=$?
  fi
  [[ -f "$output" ]] || refuse "row produced no classified result: $output" 66
  LAST_ROW_STATUS=$("$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = "GPU, XLA transcendentals — not the certified bit-exact policy"
assert row["measurement_policy_label"] == expected, row
assert "transcendentals=\x27native\x27" in row["precision_policy"], row
status = row.get("status")
assert status in {"CONFIRMED", "REFUTED", "VACUOUS", "RETRACTED"}, row
print(status)
' "$output")
  accept_classified_exit "$LAST_ROW_STATUS" "$rc" row
  printf 'SWG_GRADGATE_GPU_ROW status=%s output=%s\n' \
    "$LAST_ROW_STATUS" "$output"
}

run_suite() {
  local output=$1
  local steps=$2
  shift 2
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  local rc=0
  if invoke_suite_gate "$output" "$steps" "$@"; then
    rc=0
  else
    rc=$?
  fi
  [[ -f "$output" ]] || refuse \
    "resource suite produced no classified result at ${steps} steps" 66
  LAST_SUITE_STATUS=$("$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = row.get("status")
assert status in {"CONFIRMED", "REFUTED", "VACUOUS", "RETRACTED"}, row
print(status)
' "$output")
  accept_classified_exit "$LAST_SUITE_STATUS" "$rc" suite
  printf 'SWG_GRADGATE_GPU_SUITE steps=%s status=%s output=%s\n' \
    "$steps" "$LAST_SUITE_STATUS" "$output"
}

if [[ ${1:-} == --self-test ]]; then
  test_dir=$(mktemp -d)
  trap 'rm -rf -- "$test_dir"' EXIT
  observed=()
  for status in REFUTED VACUOUS RETRACTED CONFIRMED; do
    SWG_GRADGATE_SELF_TEST_STATUS=$status
    run_row "$test_dir/row_${status}.json" self-test
    [[ $LAST_ROW_STATUS == "$status" ]]
    run_suite "$test_dir/suite_${status}.json" 24 self-test
    [[ $LAST_SUITE_STATUS == "$status" ]]
    observed+=("$status")
  done
  [[ ${observed[*]} == "REFUTED VACUOUS RETRACTED CONFIRMED" ]] || \
    refuse "classified rows did not all continue" 66
  printf 'SWG_GRADGATE_GPU_RUNNER_SELF_TEST statuses=%s\n' "${observed[*]}"
  exit 0
fi

[[ -x "$PYTHON" ]] || refuse "missing Python environment: $PYTHON"
[[ -f "$GATE" ]] || refuse "run from the legoESM repository root; missing $GATE"
[[ -d "$EVIDENCE_DIR" ]] || refuse "evidence directory does not exist: $EVIDENCE_DIR"
[[ -z "$(git status --porcelain)" ]] || refuse "producer worktree is not clean"
readonly COMMIT=$(git rev-parse HEAD)

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env -u XLA_PYTHON_CLIENT_ALLOCATOR \
  "$PYTHON" -c 'import jax; devices=jax.devices(); assert jax.default_backend() == "gpu", jax.default_backend(); assert len(devices) == 1, devices; print(devices[0])'
nvidia-smi -i 1 --query-gpu=index,memory.used --format=csv,noheader

for steps in 1 24 720 1440; do
  run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
    agreement --card SWG --steps "$steps"
done

for steps in 24 720 1440; do
  for schedule in none uniform binomial; do
    run_row "$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json" \
      resources --card SWG --steps "$steps" --schedule "$schedule"
  done
  rows=(
    --row "$EVIDENCE_DIR/gpu1_resources_none_${steps}step.json"
    --row "$EVIDENCE_DIR/gpu1_resources_uniform_${steps}step.json"
    --row "$EVIDENCE_DIR/gpu1_resources_binomial_${steps}step.json"
  )
  run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step.json" \
    "$steps" "${rows[@]}"
  control_status=$LAST_SUITE_STATUS
  plant_output="$EVIDENCE_DIR/gpu1_resources_suite_${steps}step_plant.json"
  run_suite "$plant_output" "$steps" "${rows[@]}" \
    --plant uniform_label_uses_none
  plant_status=$LAST_SUITE_STATUS
  if [[ $control_status == CONFIRMED && $plant_status != REFUTED ]]; then
    refuse "resource-order plant did not flip ${steps}-step suite" 66
  fi
  if [[ $control_status == REFUTED && $plant_status == CONFIRMED ]]; then
    refuse "resource-order plant resurrected ${steps}-step refutation" 66
  fi
done

printf 'SWG_GRADGATE_GPU_ROWS_COMPLETE commit=%s evidence=%s\n' \
  "$COMMIT" "$EVIDENCE_DIR"
