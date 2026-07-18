#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --job-name=vtx_att2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/vtx_att2_%j.log

# Round 2: X4 dvec, X5 ascalar, X6 metrics+dvec combo.
# Reference: full 23.70/4.08; -metrics 19.64/3.71; interim 16.93/3.91.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM_fv3grid
OUT=/burg-archive/glab/users/pg2328/fv3_recon
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
cd "$REPO"

run_arm () {
  local name="$1"; local excl="$2"
  echo "=== ARM $name (ext_exclude=$excl) ==="
  $PY scripts/validate/fv3_native/run_duo_stepper_w2.py \
      --n 24 --dt 450 --days 5 --ext-bundle --ext-exclude "$excl" \
      --out "$OUT/w2_duo_stepper_c24_ext_no_${name}.npz"
  $PY scripts/validate/fv3_native/w2_duo_oracle_gate.py \
      "$OUT/w2_duo_stepper_c24_ext_no_${name}.npz" --res 24 \
      | grep -aE "run:|GATE" || true
}

run_arm dvec         dvec
run_arm ascalar      ascalar
run_arm metdvec      metrics,dvec

echo VTX_ATT2_DONE
