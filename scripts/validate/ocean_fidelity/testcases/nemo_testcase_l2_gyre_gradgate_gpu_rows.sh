#!/usr/bin/env bash
set -Eeuo pipefail

trap 'rc=$?; printf "REFUSE gyre-gradgate-gpu: command failed at line %s (exit %s)\n" "$LINENO" "$rc" >&2' ERR

readonly PYTHON=/home/dbalwada/legoESM/.venv/bin/python
readonly EVIDENCE_DIR=${1:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round5}
readonly GATE=scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py
readonly MODEL_PATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src

refuse() {
  printf 'REFUSE gyre-gradgate-gpu: %s\n' "$1" >&2
  exit "${2:-64}"
}

[[ -x "$PYTHON" ]] || refuse "missing Python environment: $PYTHON"
[[ -f "$GATE" ]] || refuse "run from the legoESM repository root; missing $GATE"
[[ -d "$EVIDENCE_DIR" ]] || refuse "evidence directory does not exist: $EVIDENCE_DIR"
[[ -z "$(git status --porcelain)" ]] || refuse "producer worktree is not clean"
readonly COMMIT=$(git rev-parse HEAD)

CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  "$PYTHON" -c 'import jax; devices=jax.devices(); assert jax.default_backend() == "gpu", jax.default_backend(); assert len(devices) == 1, devices; print(devices[0])'
nvidia-smi -i 1 --query-gpu=index,memory.used --format=csv,noheader

run_row() {
  local output=$1
  shift
  [[ ! -e "$output" ]] || refuse "refusing to overwrite evidence: $output" 65
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 PYTHONPATH="$MODEL_PATH" \
    "$PYTHON" "$GATE" "$@" --transcendentals native \
      --expect-commit "$COMMIT" --output "$output"
  "$PYTHON" -c '
import json
import pathlib
import sys

row = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = "GPU, XLA transcendentals — not the certified bit-exact policy"
assert row["measurement_policy_label"] == expected, row
assert "transcendentals=\x27native\x27" in row["precision_policy"], row
' "$output"
}

for steps in 1 6; do
  run_row "$EVIDENCE_DIR/gpu1_agreement_${steps}step.json" \
    agreement --steps "$steps"
done

for schedule in none uniform binomial; do
  for steps in 1 6; do
    run_row "$EVIDENCE_DIR/gpu1_resources_${schedule}_${steps}step.json" \
      resources --steps "$steps" --schedule "$schedule"
  done
done

printf 'GYRE_GRADGATE_GPU_ROWS_COMPLETE commit=%s evidence=%s\n' \
  "$COMMIT" "$EVIDENCE_DIR"
