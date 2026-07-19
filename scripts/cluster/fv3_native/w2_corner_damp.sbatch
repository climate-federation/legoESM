#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=w2_cd
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/w2_cd_%j.log

# W2 corner-damp arms: interior damp_v lowered, full 0.030 kept at the
# 8 cube-vertex zones.  Judge: 5-day v_ll_Linf (baseline 0.541) + L2 +
# W5/W6 stability.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/w2_cd_sweep
rm -rf "$BASE" && mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local dv="$2"; local cdv="$3"
  echo "=== ARM $name (damp_v=$dv corner_damp_v=$cdv) ==="
  LEGOESM_SW_DAMP_V=$dv LEGOESM_SW_CORNER_DAMP_V=$cdv \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test "williamson" --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|BLOWUP" | tail -6
}

run_arm W0_base          0.030 0.0
run_arm W1_dv015_cd030   0.015 0.030
run_arm W2_dv0075_cd030  0.0075 0.030

echo "=== SUMMARY (baseline v_ll 0.541) ==="
for a in W0_base W1_dv015_cd030 W2_dv0075_cd030; do
  echo "[$a] $(grep -aoE 'williamson2.*v_ll_Linf=[0-9.e+-]+' "$BASE/$a.log" | tail -1)"
  echo "      W5 $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep -a williamson5 | tail -1 | tr -s ' ' | cut -c1-60)"
  echo "      W6 $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep -a williamson6 | tail -1 | tr -s ' ' | cut -c1-60)"
done
echo W2_CD_DONE
