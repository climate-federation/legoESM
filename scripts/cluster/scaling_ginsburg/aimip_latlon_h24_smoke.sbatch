#!/bin/bash
# 24 h SUPERVISION SMOKE — does the lane actually train at the raised horizon?
#
# The training horizon moved 6 h -> 24 h on 2026-09-18 after re-measuring the
# adjoint (jobs 9829660 / 9829661: no NaN at 144 steps, bounded adjoint on the
# current scheme set).  Wiring is pinned by tests/unit/test_aimip_latlon_horizons
# but wiring is not evidence that it RUNS: the rollout is now 144 steps instead
# of 36, the loader fetches four extra ERA5 snapshots per window to reach the
# 24 h target, and the backward pass is four times deeper.  This runs the thing.
#
# Evaluation stays at 6 h on purpose (scorecard parity), so the printed RMSEs
# remain comparable to every earlier smoke.
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=aimip_h24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# 4x the rollout depth on top of the heavy rrtmgp compile.
#SBATCH --time=08:00:00
#SBATCH --output=results/aimip_latlon/logs/%x_%j.out

set -uo pipefail
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$REPO/.jax_cache_adjoint}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR" results/aimip_latlon/logs

echo "=== git SHA: $(git -C "$REPO" rev-parse HEAD)"
echo "=== horizon wiring gate ==="
"$PY_GPU" -m pytest tests/unit/test_aimip_latlon_horizons.py \
                    tests/unit/test_cloud_tuning_overrides.py \
                    tests/unit/test_grad_horizon.py -q 2>&1 | tail -6
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "HORIZON/INSTRUMENT GATE FAILED (rc=$rc) — not launching the smoke."
  exit "$rc"
fi

OUT="results/aimip_latlon/h24_smoke_${SLURM_JOB_ID}"
"$PY_GPU" -u scripts/run/run_aimip_latlon.py \
  --smoke --variants classical,column_nn --output-dir "$OUT"
rc=$?

echo "===== scorecard (eval still at 6 h) ====="
cat "$OUT/aimip_latlon_scorecard.json" 2>/dev/null \
  || echo "(no scorecard — check errors above)"
exit "$rc"
