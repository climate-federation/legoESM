#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=sw_imprint
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/sw_imprint_%j.log
set -uo pipefail   # NOT -e: rows are allowed to fail; we aggregate below

# Phase-4c cube-imprint battery.  Williamson-2 (analytic v == 0 everywhere,
# so any lat-lon-regridded v signal is PURE grid-imprint error).  Five cube
# configs + fixed cross-grid references:
#   A  production A-L, legacy equiangular + no duo   (legoESM tuned default)
#   B  production A-L, FV3-native ED + duo
#   C  FB core,        legacy equiangular + duo
#   D  FB core,        FV3-native ED + duo
#   E  FB core,        FV3-native ED + duo + native seam angles
#
# CONTROLLED-COMPARISON HONESTY (codex p4c FB-review P1/P1b): this is NOT a
# clean single-factor ladder.  NO rung isolates a single METRIC:
#   C -> D : the whole ED-native grid config (ED gnomonic metrics AND their
#            mandatory k2e_nord=4 duo tables — ED forces order 4, order-2 is
#            the wrong interpolant, ~4e-2 err; cubed_sphere.py:462), same FB
#            core + M1 config.  Closest to an ED-only rung, but still bundled.
#   D -> E : native cross-face seam angles (closest to a seam-angle-only rung)
# A->B bundles ED geometry AND no-duo->duo; B->C changes grid, solver
# (A-L->FB) AND calibration together.  Treat every rung as a whole-CONFIG
# comparison (directional evidence), not per-knob attribution.  So read
# grid/angle effects only
# from C->D and D->E; A/B/C are different-solver baselines, not rungs.
# E is 'native ED + native-angles FB experiment', NOT the fully Fortran-
# faithful FV3 config: the FB d_sw5 cross-face halo is the stable zero-ring
# approximation (the faithful ghost destabilizes; fv3_sw_core.py:3205).
# lat-lon + MPAS(icosahedral) + spectral (in A) are the fixed references.

REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon/sw_imprint
rm -rf "$OUT" && mkdir -p "$OUT"/{A_prod_legacy,B_prod_ed,C_fb_legacy,D_fb_ed,E_fb_ed_angles}

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
M="$P scripts/matrix/run_atmosphere_test_matrix.py --only sw --test williamson2"
cd "$REPO"

FAILED=""
run () { local label="$1"; local dir="$2"; shift 2
  echo; echo "===== $label ====="
  if ! $M "$@" -o "$dir" 2>&1 | tee "$dir/run.log"; then
    echo "  ROW FAILED: $label"; FAILED="$FAILED $label"
  fi
}

run "A  production A-L, legacy equiangular (+ latlon/mpas/spectral refs)" \
    "$OUT/A_prod_legacy" --grid all
run "B  production A-L, FV3-native ED + duo" \
    "$OUT/B_prod_ed"     --grid cubed_sphere --fv3-native-grid
run "C  FB core, legacy equiangular + duo" \
    "$OUT/C_fb_legacy"   --grid cubed_sphere --sw-core fb
run "D  FB core, FV3-native ED + duo" \
    "$OUT/D_fb_ed"       --grid cubed_sphere --sw-core fb --fv3-native-grid
run "E  FB core, FV3-native ED + duo + native seam angles" \
    "$OUT/E_fb_ed_angles" --grid cubed_sphere --sw-core fb --fv3-native-grid --fv3-native-angles

echo
echo "=================================================================="
echo " W2 IMPRINT TABLE  (v-wind == 0 exact; v_ll_Linf = pure imprint)"
echo " whole-CONFIG rungs (NOT single-metric): C->D (ED-native config, incl."
echo " mandatory k2e_nord=4), D->E (native seam angles) — directional only"
echo "=================================================================="
for row in A_prod_legacy B_prod_ed C_fb_legacy D_fb_ed E_fb_ed_angles; do
  echo "--- $row ---"
  grep -hiE "cubed_sphere .*shallow_water .*williamson2|v_ll_Linf" "$OUT/$row/run.log" 2>/dev/null \
    | grep -viE "^\s*#|Warning" | tail -2 || echo "  (no result — row failed)"
done
echo "--- cross-grid refs (from A) ---"
grep -hiE "(latlon|icosahedral|spectral) .*shallow_water .*williamson2" "$OUT/A_prod_legacy/run.log" 2>/dev/null \
  | grep -viE "Warning" || true

if [ -n "$FAILED" ]; then
  echo "SW_IMPRINT_BATTERY_DONE_WITH_FAILURES:$FAILED"
  exit 1
fi
echo "SW_IMPRINT_BATTERY_DONE"
