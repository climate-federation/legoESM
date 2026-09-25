#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=adj_horizon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:45:00
#SBATCH --output=results/adjoint_horizon_sweep/logs/%x_%j.out
#
# Usage: sbatch adjoint_horizon_sweep.sbatch <variant>
#   variant = dycore    (cheap: the neural step has no radiation, so the
#                        backward graph is small; this is the #797 case)
#           = classical (expensive: rrtmgp reverse-mode dominates compile)
# Split by variant on purpose.  Every horizon is a separate static scan length
# and therefore a separate compile, so one job covering both variants, both
# friction arms, both differentiation targets and both scheme sets would be
# ~96 cold compiles of a graph whose classical arm is documented at hours each.
set -uo pipefail
VARIANT="${1:-dycore}"
source "$SLURM_SUBMIT_DIR/scripts/cluster/scaling_ginsburg/_env.sh"
cd "$REPO"
mkdir -p results/adjoint_horizon_sweep/logs

export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
# Persistent compile cache: the sweep recompiles per horizon by construction,
# and reruns/arms share most graphs.  _env.sh leaves this empty (cold compiles
# are what a scaling BENCHMARK wants); this is not a benchmark.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$REPO/.jax_cache_adjoint}"
mkdir -p "$LEGOESM_JIT_CACHE_DIR"
# NOT JAX_DEBUG_NANS: this sweep MEASURES where the NaN appears and must run to
# completion rather than trap on the first one.  Trapping is a separate job.

echo "=== git SHA: $(git -C "$REPO" rev-parse HEAD)  variant=$VARIANT"
echo "=== instrument self-test (the sweep is only as good as these) ==="
"$PY_GPU" -m pytest tests/unit/test_grad_horizon.py tests/unit/test_cloud_tuning_overrides.py -q 2>&1 | tail -5
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "INSTRUMENT SELF-TEST FAILED (rc=$rc) — refusing to run the sweep."
  exit "$rc"
fi

# TWO scheme sets, measured separately and never pooled:
#   receipt — what the June 2026 NaN was actually observed under.  Recovered
#             from the receipt's own commit (24abc4949, 2026-06-20): gray
#             radiation, SBM convection, Louis turbulence, NO microphysics.
#             All four matter; dropping the microphysics flag would silently
#             run Kessler and the arm would no longer be the receipt.
#   today   — the launcher's current defaults (rrtmgp + mass flux + kessler),
#             the configuration any decision about training horizons applies to.
# Different physics.  A blow-up horizon from one says nothing about the other.
status=0
for cfg in receipt today; do
  case "$cfg" in
    receipt) EXTRA="--convection sbm --radiation gray --microphysics none" ;;
    today)   EXTRA="" ;;
  esac
  echo "############ variant=$VARIANT config=$cfg  cli-args='$EXTRA' ############"
  "$PY_GPU" -u scripts/validate/adjoint_horizon_sweep.py \
    --variants "$VARIANT" \
    --wrt ic params \
    --n-lat 32 --n-lev 8 \
    --horizons 1 2 8 36 72 144 \
    --cli-args "$EXTRA" \
    --out "results/adjoint_horizon_sweep/run_${SLURM_JOB_ID}_${VARIANT}_${cfg}"
  rc=$?
  echo "---- variant=$VARIANT config=$cfg exit=$rc ----"
  # A failed arm must not be hidden by a later successful one: an unnoticed
  # non-zero exit here is how a half-run sweep gets read as a complete result.
  [ "$rc" -ne 0 ] && status=$rc
done
exit "$status"
