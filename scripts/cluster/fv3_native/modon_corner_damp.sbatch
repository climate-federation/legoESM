#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=modon_cd
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_cd_%j.log

# Corner-localized vorticity damping arms: interior keeps a LOW damp_v
# (core preservation), the 8 cube-vertex zones keep the FULL 0.010
# (vertex-mode suppression measured necessary in the V-sweep).
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab
mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local dv="$2"; local cdv="$3"
  echo "=== ARM $name (damp_v=$dv corner_damp_v=$cdv) ==="
  LEGOESM_SW_MODON_DAMP_V=$dv LEGOESM_SW_MODON_CORNER_DAMP_V=$cdv \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test colliding_modons --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|BLOWUP" | tail -3
}

run_arm C1_dv0.0025_cd0.010  0.0025 0.010
run_arm C2_dv0.0_cd0.010     0.0    0.010

echo "=== SUMMARY ==="
for a in C1_dv0.0025_cd0.010 C2_dv0.0_cd0.010; do
  echo "[$a] $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep -a collid | tail -1)"
done
echo MODON_CD_DONE
