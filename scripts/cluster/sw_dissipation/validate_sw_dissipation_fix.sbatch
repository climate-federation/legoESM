#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=sw_dissfix
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/burg-archive/glab/users/pg2328/sw_dissfix_%j.log
set -uo pipefail

# Validation battery for the SW dissipation fixes:
#   latlon:   Laplacian A_h (~1e6 m^2/s, tau ~ 1-6 d at physical scales)
#             -> scale-selective biharmonic nu_del4 with per-row pole cap
#   spectral: per-step order-8 exp filter (e^-16 at n=10 over 100 d)
#             -> one-time IC filter (incl. winds) for colliding modons
#
# Stage 1: unit tests (new biharmonic + filter-mode tests + neighbors).
# Stage 2: full SW battery, all four grids, fresh output dir — regenerates
#          the cross-grid comparison PNGs for visual A/B against the
#          2026-07-17 baseline (W5 wind_speed, W2 v, W6 height,
#          colliding_modons vorticity).

REPO=/burg-archive/glab/users/pg2328/legoESM
OUT=$REPO/results/atmosphere_swfix_ship
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src:$REPO
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

echo "===== Stage 1: unit tests ====="
$P -m pytest -x -q \
  tests/atmosphere/shallow_water/unit/test_shallow_water_latlon_cgrid.py \
  tests/atmosphere/shallow_water/unit/test_spectral_sw_state_truncation.py \
  tests/atmosphere/shallow_water/unit/test_spectral.py \
  tests/atmosphere/shallow_water/unit/test_nesting_latlon.py \
  tests/atmosphere/shallow_water/unit/test_colliding_modons_ic.py \
  tests/atmosphere/shallow_water/unit/test_williamson_extended.py \
  2>&1 | tail -80
UNIT_RC=${PIPESTATUS[0]}
if [ "$UNIT_RC" -ne 0 ]; then
  echo "SW_DISSFIX_UNIT_TESTS_FAILED rc=$UNIT_RC"
  exit 1
fi
echo "SW_DISSFIX_UNIT_TESTS_OK"

echo
echo "===== Stage 2: SW battery, all grids ====="
rm -rf "$OUT" && mkdir -p "$OUT"
$P scripts/matrix/run_atmosphere_test_matrix.py \
  --only sw --grid all -o "$OUT" 2>&1 | tee "$OUT/run.log" | tail -60
MATRIX_RC=${PIPESTATUS[0]}

echo
echo "===== Result summary ====="
grep -iE "PASS|FAIL|blow|nan" "$OUT/run.log" | tail -40 || true
echo
echo "Cross-grid PNGs:"
find "$OUT" -name "comparison_snapshots_*.png" | sort

if [ "$MATRIX_RC" -ne 0 ]; then
  echo "SW_DISSFIX_MATRIX_FAILED rc=$MATRIX_RC"
  exit 1
fi
echo "SW_DISSFIX_DONE"
