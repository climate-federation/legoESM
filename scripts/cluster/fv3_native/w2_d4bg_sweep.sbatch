#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=w2_d4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/w2_d4_%j.log

# W2 5-day recalibration with the d_sw5 del-4 background term
# (certified d_sw5 reference; production baseline 0.541 = iters 42-71
# optimum WITHOUT d4).  Arms rebalance del-2 (div_damp factor) against
# the new del-4 (d4_bg).  Judge: W2 5-day v_ll_Linf + L2; W5/W6 run in
# the same arm as stability gates.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/w2_d4_sweep
rm -rf "$BASE" && mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local d4="$2"; local dd="$3"
  echo "=== ARM $name (d4_bg=$d4 div_damp_factor=$dd) ==="
  LEGOESM_SW_D4_BG=$d4 LEGOESM_SW_DIV_DAMP_FACTOR=$dd \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test "williamson" --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|ERROR" | tail -6
}

run_arm S0_base       0.0  8.0
run_arm S1_d416_dd8   0.16 8.0
run_arm S2_d416_dd4   0.16 4.0
run_arm S3_d416_dd2   0.16 2.0
run_arm S4_d412_dd4   0.12 4.0
run_arm S5_d420_dd4   0.20 4.0

echo "=== SUMMARY (W2 v_ll baseline 0.541; W5/W6 stability gates) ==="
for a in S0_base S1_d416_dd8 S2_d416_dd4 S3_d416_dd2 S4_d412_dd4 S5_d420_dd4; do
  echo "[$a] W2: $(grep -aoE 'williamson2.*v_ll_Linf=[0-9.e+-]+' "$BASE/$a.log" | tail -1)"
  echo "      W5: $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep williamson5 | tail -1 | tr -s ' ')"
  echo "      W6: $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep williamson6 | tail -1 | tr -s ' ')"
done
echo W2_D4_DONE
