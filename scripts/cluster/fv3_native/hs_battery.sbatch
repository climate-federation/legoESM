#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=hs_full
#SBATCH --time=11:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/hs_full_%j.log

# Phase-4c campaign EXIT battery part 2: Held-Suarez on ALL FOUR grids
# (sigma + hybrid; matrix defaults) + the cube-imprint visual-regression
# gate (the "no artifacts" component of the exit criterion).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon/hs_battery
rm -rf "$OUT" && mkdir -p "$OUT"

export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

echo "===== visual-regression cube-imprint gate (no-artifacts check) ====="
$PY scripts/validate/visual_regression.py --check 2>&1 | tee "$OUT/visreg.log"
vr_rc=$?
echo "visreg rc=$vr_rc"

echo "===== Held-Suarez, all grids ====="
$PY scripts/matrix/run_atmosphere_test_matrix.py \
  --only hydro --test held_suarez --grid all -o "$OUT" 2>&1 | tee "$OUT/run.log"
rc=$?

echo "=================================================================="
echo " HELD-SUAREZ BATTERY SUMMARY"
echo "=================================================================="
grep -E "PASS|FAIL|ERROR|SKIP" "$OUT/run.log" | grep -iE "held" | tail -20
echo "HS_BATTERY_DONE rc=$rc visreg_rc=$vr_rc"
exit $(( rc | vr_rc ))
