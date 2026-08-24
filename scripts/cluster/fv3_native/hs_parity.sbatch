#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3hs
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hs_%j.log
set -euo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/hs_$SLURM_JOB_ID
mkdir -p $OUT
cd "$REPO"
# GATED at 1e-7 (worst field v ~1.6e-8); --save-fields feeds the maps.
$PY scripts/validate/fv3_native/full_step_oracle_parity.py --physics held_suarez --max-rel 1e-7 \
    --save-fields $OUT/hs_fields.npz
$PY scripts/plot/fv3_duo_face_maps.py $OUT/hs_fields.npz --out-dir $OUT/maps
ls -la $OUT/maps/*.png
echo FV3_HS_PARITY_DONE
