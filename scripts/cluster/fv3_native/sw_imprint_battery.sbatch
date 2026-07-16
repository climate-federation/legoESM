#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=sw_imprint
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/sw_imprint_%j.log
set -euo pipefail

# Phase-4c cube-imprint A/B battery.  Williamson-2 (analytic v == 0
# everywhere, so any lat-lon-regridded v signal is PURE grid-imprint
# error).  The two cube runs differ by the WHOLE FV3-native production
# grid config: legacy (equiangular gnomonic, NO duo halos) vs FV3-native
# (ED gnomonic + duo order-4 halos).  This is NOT an isolated ED-vs-
# equiangular metric swap — the duo halo changes cross-face operator
# behaviour too (codex flag-review P1).  Held fixed across both: solver
# (A-L RK3), config (iter1009 dual-target), IC, resolution (C36), dt,
# days.  lat-lon + MPAS(icosahedral) + spectral are the fixed cross-grid
# references (unaffected by --fv3-native-grid).

REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon/sw_imprint
A=$OUT/baseline_allgrids       # equiangular cube + latlon + mpas + spectral
B=$OUT/native_cube             # FV3-native ED cube
rm -rf "$OUT" && mkdir -p "$A" "$B"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

echo "=================================================================="
echo " RUN A  baseline (equiangular cube + latlon + mpas + spectral)"
echo "=================================================================="
$P scripts/matrix/run_atmosphere_test_matrix.py \
   --only sw --grid all --test williamson2 -o "$A" 2>&1 | tee "$A/run.log"

echo "=================================================================="
echo " RUN B  FV3-native ED cube (--fv3-native-grid)"
echo "=================================================================="
$P scripts/matrix/run_atmosphere_test_matrix.py \
   --only sw --grid cubed_sphere --test williamson2 --fv3-native-grid \
   -o "$B" 2>&1 | tee "$B/run.log"

echo "=================================================================="
echo " IMPRINT A/B SUMMARY  (Williamson-2 v-wind = pure grid imprint)"
echo " A = legacy default (equiangular, no duo) | B = FV3-native (ED+duo)"
echo " Attribute any imprint change to the native CONFIG bundle, not the"
echo " grid metrics alone (duo halo also changes; codex flag-review P1)."
echo "=================================================================="
# The per-case notes line carries the regridded v-wind error norms; pull
# the williamson2 lines from each run's log so the A/B is side-by-side.
echo "--- baseline (equiangular cube / latlon / mpas / spectral) ---"
grep -iE "williamson2|v_ll|Linf|L2=" "$A/run.log" | grep -viE "^\s*#|Warning" || true
echo "--- native ED cube ---"
grep -iE "williamson2|v_ll|Linf|L2=" "$B/run.log" | grep -viE "^\s*#|Warning" || true

echo "SW_IMPRINT_BATTERY_DONE"
