#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=w2_nord
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/w2_nord_%j.log

# Williamson-2 duo-halo-order probe (user request, parallel to the modon
# sharpness sweep): cube W2 production default = NON-duogrid (0.541
# v_ll_Linf).  Arms: baseline / legacy duo order-2 / legacy duo order-4
# — same tuned solver + equiangular geometry, ONE knob (halo treatment).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/w2_nord_ab
rm -rf "$BASE" && mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local nord="$2"
  echo "=== W2 ARM $name (duo_nord=${nord:-off}) ==="
  if [ -n "$nord" ]; then export LEGOESM_SW_CUBE_DUO_NORD=$nord; else unset LEGOESM_SW_CUBE_DUO_NORD; fi
  $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test williamson2 --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|ERROR" | tail -3
}

run_arm W2_baseline ""
run_arm W2_duo_o2   2
run_arm W2_duo_o4   4

echo "=== SUMMARY (v_ll_Linf = imprint metric; baseline 0.541) ==="
for a in W2_baseline W2_duo_o2 W2_duo_o4; do
  echo "[$a] $(grep -aoE 'v_ll_Linf=[0-9.e+-]+ \(pre-regrid [0-9.e+-]+\)' "$BASE/$a.log" | tail -1)"
done
echo W2_NORD_DONE
