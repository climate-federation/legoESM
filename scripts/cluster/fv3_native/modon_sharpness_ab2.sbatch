#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=modon_ab2
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab2_%j.log

# Modon sharpness round 2: LEGACY equiangular geometry (tuned solver
# intact) + ORDER-4 duo halo tables — halo order isolated from the
# geometry swap that killed arms C/D/E.  Factor sweep on top.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab
mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local factor="$2"; local nord="$3"
  echo "=== ARM $name (hyperdiff ${factor}x k2e_nord=$nord) ==="
  LEGOESM_SW_MODON_HYPERDIFF_FACTOR=$factor \
  LEGOESM_SW_MODON_K2E_NORD=$nord \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test colliding_modons --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|ERROR" | tail -4
  echo "--- arm $name done ---"
}

run_arm F_o4_1.0   1.0  4
run_arm G_o4_0.5   0.5  4
run_arm H_o4_0.25  0.25 4

echo "=== SUMMARY ==="
for a in F_o4_1.0 G_o4_0.5 H_o4_0.25; do
  echo "[$a] $(grep -aE 'PASS|FAIL|ERROR' "$BASE/$a.log" | grep colliding | tail -1)"
done
echo MODON_AB2_DONE
