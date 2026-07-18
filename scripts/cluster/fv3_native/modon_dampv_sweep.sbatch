#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=modon_dv
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_dv_%j.log

# Modon CORE-sharpness lever: vorticity damping damp_v (del-6, nord_v=2)
# directly erodes the vortex cores; the biharmonic hyperdiff (kept at
# the stability-mandated 1.0x) guards the seams.  Sweep damp_v below
# the 0.010 sweep value on the o2-duo production config.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab
mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local dv="$2"
  echo "=== ARM $name (damp_v=$dv hyperdiff 1.0x o2-duo) ==="
  LEGOESM_SW_MODON_DAMP_V=$dv \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test colliding_modons --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|ERROR|BLOWUP" | tail -4
}

run_arm V1_dv0.005   0.005
run_arm V2_dv0.0025  0.0025
run_arm V3_dv0.0     0.0

echo "=== SUMMARY (baseline A: damp_v=0.010 PASS but cores decay 23->5) ==="
for a in V1_dv0.005 V2_dv0.0025 V3_dv0.0; do
  echo "[$a] $(grep -aE 'PASS|FAIL|BLOWUP' "$BASE/$a.log" | grep -a collid | tail -1)"
done
echo MODON_DV_DONE
