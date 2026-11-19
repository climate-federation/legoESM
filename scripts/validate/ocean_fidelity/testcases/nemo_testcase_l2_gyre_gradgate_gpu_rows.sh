#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "REFUSE ocean-gradgate-gpu: command failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2' ERR

readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py
readonly MODEL_PATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src

refuse() {
  printf 'REFUSE ocean-gradgate-gpu: %s\n' "$1" >&2
  exit "${2:-64}"
}

# One certified card per invocation; no default, so no run can silently
# measure the other card's program.
readonly CARD=${CARD:?set CARD=GYRE-zco or CARD=SWG}
case "$CARD" in
  GYRE-zco)
    DEFAULT_EVIDENCE_DIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round7
    EXPECTED_CASE=GYRE-zco
    ;;
  SWG)
    DEFAULT_EVIDENCE_DIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round3
    EXPECTED_CASE=SWG-zco
    ;;
  *) refuse "unknown card $CARD" ;;
esac
readonly EVIDENCE_DIR=${1:-$DEFAULT_EVIDENCE_DIR}
readonly EXPECTED_CASE

[[ -f "$GATE" ]] || refuse "run from the repository root; missing $GATE"
[[ -z "$(git status --porcelain)" ]] || refuse "worktree is not clean"
readonly COMMIT=$(git rev-parse HEAD)

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  env -u XLA_PYTHON_CLIENT_ALLOCATOR \
  "$PYTHON" -c 'import jax; devices=jax.devices(); assert jax.default_backend() == "gpu", jax.default_backend(); assert len(devices) == 1, devices; print(devices[0])'
nvidia-smi -i 1 --query-gpu=index,memory.used --format=csv,noheader

ROW_FAILURES=0

run_row() {
  local output=$1
  local mode=$2
  local steps=$3
  local schedule=${4:-}
  local rc=0
  local args=("$mode" --card "$CARD" --steps "$steps")
  [[ -z "$schedule" ]] || args+=(--schedule "$schedule")
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" "${args[@]}" \
        --transcendentals native \
        --expect-commit "$COMMIT" --output "$output"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE ocean-gradgate-gpu: row wrote no JSON (exit %s): %s\n' \
      "$rc" "$*" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if ! "$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
commit, card, case, mode, steps, schedule = sys.argv[2:]
expected = "GPU, XLA transcendentals — not the certified bit-exact policy"
assert row["measurement_policy_label"] == expected, row
assert "transcendentals=\x27native\x27" in row["precision_policy"], row
assert row["provenance"] == {
    "commit": commit, "dirty": False, "verified_after_measurement": True}, row
assert row["card_selector"] == card, row
assert row["case"] == case, row
assert row["backend"] == "gpu", row
assert row["physical_gpu_index"] == 1, row
assert row["cuda_visible_devices"] == "1", row
assert row["steps"] == int(steps), row
if schedule:
    assert row["schedule"] == schedule, row
if mode == "resources":
    assert row["resource_schema"] == "gyre-gradgate-resource-v2-compiled-reverse", row
    assert row["reverse_transform"] == "jax.jit(jax.value_and_grad(scalar_line))", row
    assert row["warm_reverse_timer_span"].startswith("blocked execution"), row
    assert row["xla_python_client_preallocate"].lower() == "false", row
    assert row["xla_python_client_allocator"] is None, row
elif card == "GYRE-zco":
    assert row["schedule_scope"].startswith("Decision-49 per-step"), row
else:
    assert row["composition"] == "per-step Python loop", row
' "$output" "$COMMIT" "$CARD" "$EXPECTED_CASE" "$mode" "$steps" "$schedule"; then
    printf 'REFUSE ocean-gradgate-gpu: invalid row provenance: %s\n' "$output" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if ((rc != 0)); then
    printf 'REFUSE ocean-gradgate-gpu: row exited %s after writing JSON: %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
  fi
}

run_suite() {
  # run_suite OUTPUT STEPS --row A --row B --row C [--plant NAME]
  local output=$1
  local steps=$2
  shift 2
  local plant=""
  local arg
  for arg in "$@"; do
    [[ "$arg" != --plant ]] || plant=1
  done
  local rc=0
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  if CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$MODEL_PATH" \
      env -u XLA_PYTHON_CLIENT_ALLOCATOR \
      "$PYTHON" "$GATE" resource-suite --card "$CARD" --steps "$steps" \
        --expect-commit "$COMMIT" --output "$output" "$@"; then
    rc=0
  else
    rc=$?
  fi
  if [[ ! -f "$output" ]]; then
    printf 'REFUSE ocean-gradgate-gpu: suite wrote no JSON (exit %s): %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
    return 0
  fi
  if [[ -z "$plant" && $rc -ne 0 ]]; then
    printf 'REFUSE ocean-gradgate-gpu: resource suite off-bar: %s\n' "$output" >&2
    ((ROW_FAILURES += 1))
  elif [[ -n "$plant" && $rc -ne 1 ]]; then
    printf 'REFUSE ocean-gradgate-gpu: ordering plant did not flip (exit %s): %s\n' \
      "$rc" "$output" >&2
    ((ROW_FAILURES += 1))
  fi
}

run_resource_block() {
  local steps=$1
  local schedule
  for schedule in none uniform binomial; do
    run_row "$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json" \
      resources "$steps" "$schedule"
  done
  local rows=(
    --row "$EVIDENCE_DIR/gpu1_resources_none_${steps}step.json"
    --row "$EVIDENCE_DIR/gpu1_resources_uniform_${steps}step.json"
    --row "$EVIDENCE_DIR/gpu1_resources_binomial_${steps}step.json"
  )
  run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step.json" \
    "$steps" "${rows[@]}"
  # The ordering plant must flip the suite verdict (exit 1).
  run_suite "$EVIDENCE_DIR/gpu1_resources_suite_${steps}step_plant.json" \
    "$steps" "${rows[@]}" --plant uniform_label_uses_none
}

if [[ "$CARD" == GYRE-zco ]]; then
  # GYRE round 7: agreement composes --schedule per_step explicitly.
  for steps in 1 6; do
    run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
      agreement "$steps" per_step
  done
  for steps in 1 6; do
    run_resource_block "$steps"
  done
  COMPLETE_TAG=GYRE_GRADGATE_GPU_ROWS_COMPLETE
else
  # SWG round 3: direct 1440-step rows, agreement on the default composition.
  for steps in 1 24 720 1440; do
    run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
      agreement "$steps"
  done
  for steps in 24 720 1440; do
    run_resource_block "$steps"
  done
  COMPLETE_TAG=SWG_GRADGATE_GPU_ROWS_COMPLETE
fi

((ROW_FAILURES == 0)) || refuse "$ROW_FAILURES measurement rows failed" 1

printf '%s commit=%s evidence=%s\n' \
  "$COMPLETE_TAG" "$COMMIT" "$EVIDENCE_DIR"
