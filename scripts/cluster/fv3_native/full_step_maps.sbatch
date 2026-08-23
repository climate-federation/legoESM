#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=fv3maps
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/maps_%j.log
set -euo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
OUT=/burg-archive/glab/users/pg2328/fv3_duo_gaps/maps_$SLURM_JOB_ID
mkdir -p $OUT
cd "$REPO"

for ARM in hydro nh; do
  echo "=== ARM=$ARM ==="
  case $ARM in
    hydro) FLAGS="" ; GATE=1e-8 ;;
    nh)    FLAGS="--nh --step-run /burg-archive/glab/users/pg2328/fv3_oracle_pinned/run_nh_1step_gfs" ; GATE=1e-5 ;;
  esac
  # set -e is on: a failed parity or a failed plot ENDS the job
  # rather than falling through to a successful-looking tail (codex
  # BLOCKER, 2026-08-23).
  $PY scripts/validate/fv3_native/full_step_oracle_parity.py $FLAGS \
      --max-rel $GATE --save-fields $OUT/${ARM}_fields.npz
  # The boundary numbers are REPORT-ONLY. The gate on this run is the
  # parity residual above; both reviewers of the edge metric found
  # fields it would pass while visibly wrong, so it does not gate.
  $PY scripts/plot/fv3_duo_face_maps.py $OUT/${ARM}_fields.npz \
      --out-dir $OUT/$ARM
done
ls -la $OUT/*/*.png
echo FV3_MAPS_DONE
