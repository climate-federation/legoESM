#!/usr/bin/env bash
# Cheap cubed-sphere smoke test ON a small Cloud TPU VM (v5litepod-4).
#
# Purpose: a fast, low-cost "does it run on real TPU hardware at all" check
# before committing to a full v5e-8 scaling sweep.  This is NOT a meaningful
# scaling benchmark — it uses tiny resolution and few timing iterations.
#
# On a 4-chip VM the valid cubed-sphere device counts are 1 and 2 only:
# face sharding needs a divisor of the 6-face layout, and neither 4 nor 8
# divides 6 (see docs/scaling/scaling_tpu.md).  So this sweeps 1 -> 2 chips
# (2 of the 4 chips idle) at C24/L10 with float32.
#
# Run from the repo root on the TPU VM (after setup_env.sh):
#   bash scripts/cluster/gcp_tpu/run_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cheap defaults: 2 chips max, tiny grid, short timing.  All still overridable
# (run_bench.sh reads these same env vars); we only lower the defaults here.
export N_GPUS="${N_GPUS:-2}"                       # use 2 of 4 chips: face sharding
export N_LEVELS="${N_LEVELS:-10}"
export STRONG_RESOLUTIONS="${STRONG_RESOLUTIONS:-24}"
export N_WARMUP="${N_WARMUP:-2}"
export N_TIMING="${N_TIMING:-10}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/scaling_tpu_smoke}"

echo "=== TPU SMOKE TEST (v5litepod-4: cubed-sphere, float32, 2 of 4 chips) ==="
echo

exec bash "$SCRIPT_DIR/run_bench.sh"