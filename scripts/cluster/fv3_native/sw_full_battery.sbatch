#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=sw_full
#SBATCH --time=11:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/sw_full_%j.log

# Phase-4c campaign EXIT battery: the FULL SW matrix (Williamson 2/5/6,
# cosine bells, colliding modons) on ALL FOUR grids (cubed_sphere C36 +
# latlon + icosahedral/MPAS + spectral) with the PRODUCTION configuration
# — the user's exit criterion: every case passes, no artifacts, cube close
# to the other grids.  Cross-grid comparison snapshots are produced by the
# matrix itself (comparison_snapshots_<field>.png; the W2 v-wind panel is
# the canonical cube-imprint diagnostic).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon/sw_full_battery
rm -rf "$OUT" && mkdir -p "$OUT"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd "$REPO"

/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python \
  scripts/matrix/run_atmosphere_test_matrix.py \
  --only sw --grid all -o "$OUT" 2>&1 | tee "$OUT/run.log"
rc=$?

echo "=================================================================="
echo " FULL SW BATTERY SUMMARY (exit criterion: all PASS, cube ~ others)"
echo "=================================================================="
grep -E "PASS|FAIL|ERROR|SKIP" "$OUT/run.log" | grep -E "shallow_water" | tail -40
echo "SW_FULL_BATTERY_DONE rc=$rc"
exit $rc
