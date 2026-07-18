#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=modon_or
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/modon_or_%j.log

# ORACLE-RECIPE modon arms (Zenodo case-8 duo input.nml: nord=2
# d4_bg=0.12, do_vort_damp=.false.): replace the vorticity-damping +
# wind-hyperdiff families with the structured del-6 divergence damping.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
BASE=/burg-archive/glab/users/pg2328/fv3_recon/modon_ab
mkdir -p "$BASE"
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/packages/ocean:$REPO/packages/land:$REPO/packages/seaice:$REPO/packages/ice:$REPO/packages/coupler:$REPO/packages/ml:$REPO/packages/tools:$REPO/src
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
cd "$REPO"

run_arm () {
  local name="$1"; local dv="$2"; local hd="$3"; local d4="$4"; local dd="$5"
  echo "=== ARM $name (damp_v=$dv hyperdiff=$hd d4_bg=$d4 nord2 div_damp=$dd) ==="
  LEGOESM_SW_MODON_DAMP_V=$dv \
  LEGOESM_SW_MODON_HYPERDIFF_FACTOR=$hd \
  LEGOESM_SW_MODON_D4_BG=$d4 LEGOESM_SW_MODON_D4_NORD=2 \
  LEGOESM_SW_MODON_DIV_DAMP_FACTOR=$dd \
    $PY scripts/matrix/run_atmosphere_test_matrix.py \
      --only sw --test colliding_modons --grid cubed_sphere \
      -o "$BASE/$name" 2>&1 | tee "$BASE/$name.log" \
      | grep -aE "PASS|FAIL|BLOWUP" | tail -3
}

# O1: oracle damping family ON TOP of current stability envelope
run_arm O1_d412_full   0.0 1.0 0.12 8.0
# O2: shed the wind hyperdiff (oracle has none)
run_arm O2_d412_nohd   0.0 0.0 0.12 8.0
# O3: pure oracle recipe (no del-2 aggregate either)
run_arm O3_d412_pure   0.0 0.0 0.12 0.0

echo "=== SUMMARY ==="
for a in O1_d412_full O2_d412_nohd O3_d412_pure; do
  echo "[$a] $(grep -aE 'PASS|FAIL' "$BASE/$a.log" | grep -a collid | tail -1)"
done
echo MODON_OR_DONE
