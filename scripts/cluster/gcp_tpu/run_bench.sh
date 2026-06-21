#!/usr/bin/env bash
# Run the cubed-sphere FV3 scaling benchmark ON a Cloud TPU VM.
#
# Phase 3 of the TPU plan: real-hardware scaling on a single-host slice.
#   - float32 only (TPUs have no efficient native float64; spectral excluded)
#   - cubed-sphere face sharding on 6 of the 8 chips (8 does not divide the
#     6-face layout; the 8-chip "level fallback" path is unsupported for the
#     cubed-sphere dycore — see docs/performance/scaling/scaling_tpu.md)
#
# This sweeps the valid cubed-sphere device counts (1, 2, 3, 6) at a fixed
# resolution (strong scaling).  Override knobs via env vars.
#
# Run from the repo root on the TPU VM (after setup_env.sh):
#   bash scripts/cluster/gcp_tpu/run_bench.sh
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
GRID="${GRID:-cubed-sphere}"
MODE="${MODE:-strong}"
PRECISION="${PRECISION:-float32}"
N_GPUS="${N_GPUS:-6}"            # use 6 of 8 chips: face sharding
N_LEVELS="${N_LEVELS:-26}"
STRONG_RESOLUTIONS="${STRONG_RESOLUTIONS:-48,96}"
N_WARMUP="${N_WARMUP:-3}"
N_TIMING="${N_TIMING:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-results/scaling_tpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# The bench driver imports tests.test_cases.baroclinic_wave (the shared IC),
# so the repo root must be importable.  Without this every run silently
# reports "FAILED: No module named 'tests'".
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Do NOT set JAX_PLATFORMS — let JAX pick the TPU backend.  Do NOT enable x64.

echo "TPU benchmark: grid=$GRID mode=$MODE precision=$PRECISION n_gpus=$N_GPUS"
echo "  resolutions=$STRONG_RESOLUTIONS levels=$N_LEVELS warmup=$N_WARMUP timing=$N_TIMING"
echo "  output=$OUTPUT_DIR"
echo

python scripts/bench/run_levante_gpu_scaling.py \
  --grid "$GRID" \
  --mode "$MODE" \
  --precision "$PRECISION" \
  --n-gpus "$N_GPUS" \
  --n-levels "$N_LEVELS" \
  --strong-resolutions "$STRONG_RESOLUTIONS" \
  --n-warmup "$N_WARMUP" \
  --n-timing "$N_TIMING" \
  --output-dir "$OUTPUT_DIR" \
  --no-plot

echo
echo "Results under $OUTPUT_DIR/ (copy back with gcloud compute tpus tpu-vm scp)."