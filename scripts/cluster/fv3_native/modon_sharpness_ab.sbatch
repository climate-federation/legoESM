#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=modon_ab
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab_%j.log

# Colliding-modons sharpness A/B (user: "visual issues on the colliding
# modons" — cube diffuses the vortices the ico grid preserves; the
# 2026-07-03 sweep pinned damp_v=0.010 + 1.0x hyperdiff as the MINIMUM
# STABLE damping because 0.5x erupts at the cube corners; the price is
# 23->5 m/s amplitude decay over the 100-day run).  Arms:
#   A baseline        legacy equiangular duo (k2e order-2), hyperdiff 1.0x
#   B frontier-check  legacy duo,                     hyperdiff 0.5x
#   C native o4       --fv3-native-grid (ED + certified order-4 duo), 1.0x
#   D native o4       --fv3-native-grid, 0.5x
#   E native o4       --fv3-native-grid, 0.25x
# The order-4 duo halos attack the seam-eruption mechanism directly; if
# C/D/E hold 100 d without eruption at lower hyperdiff, the modons keep
# their cores and the visual issue closes structurally.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab
rm -rf "$BASE" && mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local factor="$2"; local native="$3"
  echo "=== ARM $name (hyperdiff ${factor}x native=$native) ==="
  local flags=""
  if [ "$native" = "yes" ]; then flags="--fv3-native-grid"; fi
  LEGOESM_SW_MODON_HYPERDIFF_FACTOR=$factor \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test colliding_modons --grid cubed_sphere \
      $flags -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -E "PASS|FAIL|ERROR|max\|u\||blow|erupt" | tail -6
  echo "--- arm $name rc=$? ---"
}

run_arm A_legacy_1.0   1.0  no
run_arm B_legacy_0.5   0.5  no
run_arm C_native_1.0   1.0  yes
run_arm D_native_0.5   0.5  yes
run_arm E_native_0.25  0.25 yes

echo "=== SUMMARY ==="
for a in A_legacy_1.0 B_legacy_0.5 C_native_1.0 D_native_0.5 E_native_0.25; do
  echo "[$a] $(grep -aE 'PASS|FAIL|ERROR' "$BASE/$a.log" | grep colliding | tail -1)"
done
echo MODON_AB_DONE
