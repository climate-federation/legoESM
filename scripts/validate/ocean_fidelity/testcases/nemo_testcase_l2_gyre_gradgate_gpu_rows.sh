#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "REFUSE gyre-gradgate-gpu: command failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2' ERR

readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly EVIDENCE_DIR=${1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round9}
readonly GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py
readonly MODEL_PATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src
if [[ ${1:-} == --self-test ]]; then
  readonly SELF_TEST_MODE=1
else
  readonly SELF_TEST_MODE=0
fi
readonly -a AGREEMENT_WINDOWS=(1 6 60 360)
readonly -a RESOURCE_WINDOWS=(6 60 360)
readonly -a CHECKPOINT_SCHEDULES=(none uniform binomial)

refuse() {
  printf 'REFUSE gyre-gradgate-gpu: %s\n' "$1" >&2
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
    REFUTED|UNMEASURED|VACUOUS|RETRACTED)
      [[ $rc -eq 1 ]] || refuse "$status $label exited $rc" "$rc"
      ;;
    *)
      refuse "unknown $label status: $status" 66
      ;;
  esac
}

if [[ $SELF_TEST_MODE -eq 1 ]]; then
  [[ ${AGREEMENT_WINDOWS[*]} == "1 6 60 360" ]] || \
    refuse "agreement window registry changed" 66
  [[ ${RESOURCE_WINDOWS[*]} == "6 60 360" ]] || \
    refuse "resource window registry changed" 66
  [[ ${CHECKPOINT_SCHEDULES[*]} == "none uniform binomial" ]] || \
    refuse "checkpoint schedule registry changed" 66
  observed=()
  for status in REFUTED UNMEASURED VACUOUS RETRACTED CONFIRMED; do
    if [[ $status == CONFIRMED ]]; then
      rc=0
    else
      rc=1
    fi
    accept_classified_exit "$status" "$rc" self-test
    observed+=("$status")
  done
  [[ ${observed[*]} == \
    "REFUTED UNMEASURED VACUOUS RETRACTED CONFIRMED" ]] || \
    refuse "classified rows did not all continue" 66
  printf 'GYRE_GRADGATE_GPU_RUNNER_SELF_TEST statuses=%s\n' "${observed[*]}"
  exit 0
fi

[[ -x "$PYTHON" ]] || refuse "missing Python environment: $PYTHON"
[[ -f "$GATE" ]] || refuse "run from the legoESM repository root; missing $GATE"
mkdir -p -- "$EVIDENCE_DIR"
[[ -d "$EVIDENCE_DIR" ]] || refuse "could not create evidence directory: $EVIDENCE_DIR"
[[ -z "$(git status --porcelain)" ]] || refuse "producer worktree is not clean"
readonly COMMIT=$(git rev-parse HEAD)

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env -u XLA_PYTHON_CLIENT_ALLOCATOR \
  "$PYTHON" -c 'import jax; devices=jax.devices(); assert jax.default_backend() == "gpu", jax.default_backend(); assert len(devices) == 1, devices; print(devices[0])'
nvidia-smi -i 1 --query-gpu=index,memory.used --format=csv,noheader

ROW_FAILURES=0
LAST_ROW_CLASSIFIED=0
LAST_ROW_STATUS=

run_row() {
  local output=$1
  local mode=$2
  local steps=$3
  local schedule=$4
  local agreement_row=${5:-}
  local plant=${6:-}
  local rc=0
  LAST_ROW_CLASSIFIED=0
  LAST_ROW_STATUS=
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  args=("$mode" --steps "$steps" --schedule "$schedule" \
    --transcendentals native --expect-commit "$COMMIT" --output "$output")
  [[ -z "$agreement_row" ]] || args+=(--agreement-row "$agreement_row")
  [[ -z "$plant" ]] || args+=(--plant "$plant")
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" "${args[@]}"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE gyre-gradgate-gpu: row wrote no JSON (exit %s): %s %s %s\n' \
      "$rc" "$mode" "$steps" "$schedule" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if ! LAST_ROW_STATUS=$("$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
commit, mode, steps, schedule, agreement_row, plant = sys.argv[2:]
expected = "GPU, XLA transcendentals — not the certified bit-exact policy"
assert row["measurement_policy_label"] == expected, row
assert "transcendentals=\x27native\x27" in row["precision_policy"], row
assert row["provenance"] == {
    "commit": commit, "dirty": False, "verified_after_measurement": True}, row
assert row["case"] == "GYRE-zco", row
assert row["backend"] == "gpu", row
assert row["physical_gpu_index"] == 1, row
assert row["cuda_visible_devices"] == "1", row
assert row["steps"] == int(steps), row
assert row["schedule"] == schedule, row
status = row.get("status")
assert status in {
    "CONFIRMED", "REFUTED", "UNMEASURED", "VACUOUS", "RETRACTED"}, row
if mode == "resources":
    assert row["resource_schema"] == "gyre-gradgate-resource-v4-transformed-primal-byte-gate", row
    assert row["reverse_transform"] == "jax.jit(jax.value_and_grad(loss_with_final_state, has_aux=True))", row
    assert row["warm_reverse_timer_span"].startswith("blocked execution"), row
    assert row["xla_python_client_preallocate"].lower() == "false", row
    assert row["xla_python_client_allocator"] is None, row
    assert row["cold_first_step_segmented"] is True, row
    assert row["checkpointed_warm_steps"] == int(steps) - 1, row
    assert row["checkpoint_schedule_first_model_step"] == 2, row
    assert row["schedule_primal_bitwise_production"] is True, row
    assert isinstance(row["reverse_primal_bitwise_production"], bool), row
    assert row["resource_status"] in {"CONFIRMED", "REFUTED", "VACUOUS", "RETRACTED"}, row
    assert row["gradient_agreement_status"] in {"CONFIRMED", "REFUTED"}, row
    for key in ("reverse", "forward", "finite_difference",
                "rev_fwd_relative_error", "rev_fd_relative_error"):
        assert isinstance(row[key], (int, float)), (key, row)
    assert row["agreement_reference"] == agreement_row, row
else:
    footprint = row["step_transform_compile_footprint"]
    assert footprint["whole_window_transform_jitted"] is False, row
    assert footprint["distinct_state_pytree_specializations"] == (
        1 if int(steps) == 1 else 2), row
    assert footprint["vjp_step_invocations"] == int(steps), row
    assert footprint["jvp_step_invocations"] == int(steps), row
    assert row["step_boundary_primal_source"] == (
        "separately materialized production _step_jitted output"), row
    if plant == "boundary_primal":
        assert status == "UNMEASURED", row
        assert row["step_vjp_boundary_outputs_bitwise"] is False, row
        assert row["step_jvp_boundary_outputs_bitwise"] is False, row
    else:
        assert row["step_vjp_boundary_outputs_bitwise"] is True, row
        assert row["step_jvp_boundary_outputs_bitwise"] is True, row
print(status)
' "$output" "$COMMIT" "$mode" "$steps" "$schedule" "$agreement_row" "$plant"); then
    printf 'REFUSE gyre-gradgate-gpu: invalid row provenance: %s\n' \
      "$output" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  LAST_ROW_CLASSIFIED=1
  accept_classified_exit "$LAST_ROW_STATUS" "$rc" row
  printf 'GYRE_GRADGATE_GPU_ROW mode=%s steps=%s schedule=%s status=%s output=%s\n' \
    "$mode" "$steps" "$schedule" "$LAST_ROW_STATUS" "$output"
  if [[ $plant == boundary_primal ]]; then
    [[ $LAST_ROW_STATUS == UNMEASURED ]] || ((ROW_FAILURES += 1))
  elif [[ $LAST_ROW_STATUS != CONFIRMED ]]; then
    ((ROW_FAILURES += 1))
  fi
}

run_suite() {
  local output=$1
  local mode=$2
  local plant=${3:-}
  shift 3 || true
  local rc=0
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  args=("$mode" --expect-commit "$COMMIT" --output "$output")
  for row in "$@"; do
    args+=(--input-row "$row")
  done
  [[ -z "$plant" ]] || args+=(--plant "$plant")
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" "${args[@]}"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE gyre-gradgate-gpu: suite wrote no JSON (exit %s): %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
    LAST_SUITE_STATUS=UNMEASURED
    return 0
  fi
  LAST_SUITE_STATUS=$("$PYTHON" -c '
import json
import pathlib
import sys
row = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = row.get("status")
assert status in {"CONFIRMED", "REFUTED", "UNMEASURED", "VACUOUS", "RETRACTED"}, row
print(status)
' "$output") || refuse "invalid suite JSON: $output" 66
  accept_classified_exit "$LAST_SUITE_STATUS" "$rc" suite
  printf 'GYRE_GRADGATE_GPU_SUITE status=%s output=%s\n' \
    "$LAST_SUITE_STATUS" "$output"
}

short_rows_classified=1
for steps in "${AGREEMENT_WINDOWS[@]:0:3}"; do
  run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
    agreement "$steps" per_step
  if [[ $LAST_ROW_CLASSIFIED -ne 1 ]]; then
    short_rows_classified=0
  fi
done
run_row "$EVIDENCE_DIR/gpu1_agreement_1step_boundary_plant.json" \
  agreement 1 per_step "" boundary_primal
if [[ $short_rows_classified -eq 1 ]]; then
  long_steps=${AGREEMENT_WINDOWS[3]}
  run_row "$EVIDENCE_DIR/gpu1_agreement_${long_steps}step.json" \
    agreement "$long_steps" per_step
else
  printf 'REFUSE gyre-gradgate-gpu: 360-step row skipped because a shorter agreement row produced no classified JSON\n' >&2
  ((ROW_FAILURES += 1))
fi

for steps in "${RESOURCE_WINDOWS[@]}"; do
  resource_rows=()
  for schedule in "${CHECKPOINT_SCHEDULES[@]}"; do
    output="$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json"
    run_row "$output" resources "$steps" "$schedule" \
      "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json"
    if [[ $LAST_ROW_CLASSIFIED -eq 1 ]]; then
      resource_rows+=("$output")
    fi
  done
  if [[ ${#resource_rows[@]} -eq 3 && $steps -eq 360 ]]; then
    run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step.json" \
      gpu-resource-suite "" "${resource_rows[@]}"
    control_status=$LAST_SUITE_STATUS
    if [[ $control_status != CONFIRMED ]]; then
      ((ROW_FAILURES += 1))
    fi
    run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step_plant.json" \
      gpu-resource-suite gpu_ordering "${resource_rows[@]}"
    plant_status=$LAST_SUITE_STATUS
    if [[ $control_status == CONFIRMED && $plant_status != REFUTED ]]; then
      printf 'REFUSE gyre-gradgate-gpu: ordering plant did not flip confirmed suite\n' >&2
      ((ROW_FAILURES += 1))
    fi
    if [[ $control_status == REFUTED && $plant_status == CONFIRMED ]]; then
      printf 'REFUSE gyre-gradgate-gpu: ordering plant resurrected refuted suite\n' >&2
      ((ROW_FAILURES += 1))
    fi
  elif [[ ${#resource_rows[@]} -ne 3 ]]; then
    printf 'REFUSE gyre-gradgate-gpu: %s-step resource suite skipped because only %s of 3 rows produced classified JSON\n' \
      "$steps" "${#resource_rows[@]}" >&2
    ((ROW_FAILURES += 1))
  fi
done

memory_rows=()
for schedule in "${CHECKPOINT_SCHEDULES[@]}"; do
  for steps in "${RESOURCE_WINDOWS[@]}"; do
    row="$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json"
    [[ -f "$row" ]] && memory_rows+=("$row")
  done
done
if [[ ${#memory_rows[@]} -eq 9 ]]; then
  run_suite "$EVIDENCE_DIR/gpu1_memory_scaling.json" \
    memory-scaling-suite "" "${memory_rows[@]}"
  memory_status=$LAST_SUITE_STATUS
  [[ $memory_status == CONFIRMED ]] || ((ROW_FAILURES += 1))
  run_suite "$EVIDENCE_DIR/gpu1_memory_scaling_plant.json" \
    memory-scaling-suite flat_none_memory "${memory_rows[@]}"
  memory_plant_status=$LAST_SUITE_STATUS
  if [[ $memory_status == CONFIRMED && $memory_plant_status != REFUTED ]]; then
    printf 'REFUSE gyre-gradgate-gpu: flat-memory plant did not flip scaling suite\n' >&2
    ((ROW_FAILURES += 1))
  fi
else
  printf 'REFUSE gyre-gradgate-gpu: memory scaling suite has only %s of 9 rows\n' \
    "${#memory_rows[@]}" >&2
  ((ROW_FAILURES += 1))
fi

for schedule in "${CHECKPOINT_SCHEDULES[@]}"; do
  scaling_rows=()
  for steps in "${RESOURCE_WINDOWS[@]}"; do
    row="$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json"
    [[ -f "$row" ]] && scaling_rows+=("$row")
  done
  if [[ ${#scaling_rows[@]} -eq 3 ]]; then
    run_suite "$EVIDENCE_DIR/gpu1_time_scaling_${schedule}.json" \
      time-scaling-suite "" "${scaling_rows[@]}"
    scaling_status=$LAST_SUITE_STATUS
    [[ $scaling_status == CONFIRMED ]] || ((ROW_FAILURES += 1))
    run_suite "$EVIDENCE_DIR/gpu1_time_scaling_${schedule}_plant.json" \
      time-scaling-suite constant_time "${scaling_rows[@]}"
    scaling_plant_status=$LAST_SUITE_STATUS
    if [[ $scaling_status == CONFIRMED && $scaling_plant_status != REFUTED ]]; then
      printf 'REFUSE gyre-gradgate-gpu: constant-time plant did not flip %s scaling suite\n' \
        "$schedule" >&2
      ((ROW_FAILURES += 1))
    fi
  else
    printf 'REFUSE gyre-gradgate-gpu: %s scaling suite has only %s of 3 rows\n' \
      "$schedule" "${#scaling_rows[@]}" >&2
    ((ROW_FAILURES += 1))
  fi
done

((ROW_FAILURES == 0)) || refuse "$ROW_FAILURES measurement rows failed" 1

printf 'GYRE_GRADGATE_GPU_ROWS_COMPLETE commit=%s evidence=%s\n' \
  "$COMMIT" "$EVIDENCE_DIR"
