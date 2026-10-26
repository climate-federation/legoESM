#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cornarea
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/corner_area_%j.log
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
H=${HDUMP:-/burg-archive/glab/users/pg2328/fv3_wsubstep/run_nh_1step}
ls "$H"/hdump1_pre_zh1_t1.dat >/dev/null 2>&1 || { echo "PREFLIGHT FAILED: no height dumps under $H"; exit 4; }
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
$PY scripts/validate/fv3_native/nh_corner_area_arm.py --on --full-step --hdump "$H"
rc=$?
echo "  rc=$rc"
[ "$rc" -eq 0 ] || { echo "CORNER_AREA_FAILED rc=$rc"; exit "$rc"; }
echo CORNER_AREA_DONE
