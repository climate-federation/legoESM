#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=vtx_attr
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --exclude=g161,g203
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_recon/vtx_attr_%j.log

# C24 vertex piecewise attribution (oracle-closeness brick 3): the full
# ext bundle scores 23.70/4.08 (max at cube VERTICES) vs interim
# 16.93/3.91 (max at edge-midlat).  Swap ONE ext family at a time back
# to the interim exchange and re-score — the family whose removal
# collapses the vertex max is the driver.
#   X1 divgd    (B-scalar ext -> interim; the del6/nord damping feed)
#   X2 cvec     (C-vector ext -> interim uc/vc)
#   X3 metrics  (ext halo metrics -> kinked; ext values kept)
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
      | grep -aE "run:|envelope|GATE" || true
}

run_arm divgd   divgd
run_arm cvec    cvec
run_arm metrics metrics

echo "=== REFERENCE: full ext 23.70/4.08 (251x/106x); interim 16.93/3.91 ==="
echo VTX_ATTR_DONE
