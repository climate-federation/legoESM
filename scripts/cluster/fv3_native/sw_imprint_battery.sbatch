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
set -euo pipefail

# Phase-4c cube-imprint battery.  Williamson-2 (analytic v == 0 everywhere,
# so any lat-lon-regridded v signal is PURE grid-imprint error).  The
# "faithful FV3 ladder" — each row changes ONE thing from the row above so
# an effect can be attributed:
#   A  production A-L, legacy equiangular + no duo   (legoESM tuned default)
#   B  production A-L, FV3-native ED + duo           (A-L on native grid;
#                                                     config tuned for A)
#   C  FB core,        legacy equiangular + duo      (faithful solver, legacy grid)
#   D  FB core,        FV3-native ED + duo           (faithful solver + native metrics)
#   E  FB core,        FV3-native ED + duo + angles  (the faithful FV3 config)
# lat-lon + MPAS(icosahedral) + spectral (in A) are the fixed cross-grid
# references.  The A-L production solver's operators are tuned to the
# equiangular/legacy seam discretisation, so B is expected to be mis-
# calibrated; the FB ladder C->E is where FV3 faithfulness actually lives.

REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon/sw_imprint
rm -rf "$OUT" && mkdir -p "$OUT"/{A_prod_legacy,B_prod_ed,C_fb_legacy,D_fb_ed,E_fb_ed_angles}

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
P=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
M="$P scripts/matrix/run_atmosphere_test_matrix.py --only sw --test williamson2"
cd "$REPO"

run () { echo; echo "===== $1 ====="; shift; local dir="$1"; shift; ( $M "$@" -o "$dir" 2>&1 | tee "$dir/run.log" ) || echo "  (row failed — see log)"; }

run "A  production A-L, legacy equiangular (+ latlon/mpas/spectral refs)" \
    "$OUT/A_prod_legacy" --grid all
run "B  production A-L, FV3-native ED + duo" \
    "$OUT/B_prod_ed"     --grid cubed_sphere --fv3-native-grid
run "C  FB core, legacy equiangular + duo" \
    "$OUT/C_fb_legacy"   --grid cubed_sphere --sw-core fb
run "D  FB core, FV3-native ED + duo" \
    "$OUT/D_fb_ed"       --grid cubed_sphere --sw-core fb --fv3-native-grid
run "E  FB core, FV3-native ED + duo + native seam angles (faithful FV3)" \
    "$OUT/E_fb_ed_angles" --grid cubed_sphere --sw-core fb --fv3-native-grid --fv3-native-angles

echo
echo "=================================================================="
echo " W2 IMPRINT LADDER  (v-wind == 0 exact; v_ll_Linf = pure imprint)"
echo "=================================================================="
for row in A_prod_legacy B_prod_ed C_fb_legacy D_fb_ed E_fb_ed_angles; do
  echo "--- $row ---"
  grep -hiE "cubed_sphere .*shallow_water .*williamson2|v_ll_Linf" "$OUT/$row/run.log" \
    | grep -viE "^\s*#|Warning" | tail -2 || true
done
echo "--- cross-grid refs (from A) ---"
grep -hiE "(latlon|icosahedral|spectral) .*shallow_water .*williamson2" "$OUT/A_prod_legacy/run.log" \
  | grep -viE "Warning" || true

echo "SW_IMPRINT_BATTERY_DONE"
