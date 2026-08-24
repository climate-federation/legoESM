#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=nhtrans
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/nh_trans_%j.log
# Runs the metric transplant experiment. WIDEN=1 adds the third arm that
# also imports the corner-diagonal cells the original arm's both-sides
# sentinel mask skipped -- the region where this session measured the NH
# error to live.
set -uo pipefail
REPO=${REPO:-/burg-archive/glab/users/pg2328/legoESM_fv3jax}
[ -e "$REPO/.git" ] || { echo "PREFLIGHT FAILED: REPO=$REPO"; exit 5; }
DIRTY=$(git -C "$REPO" status --porcelain | wc -l)
[ "$DIRTY" -eq 0 ] || { echo "PREFLIGHT FAILED: dirty ($DIRTY)"; exit 3; }
echo "host: $(hostname)"; echo "PINNED sha: $(git -C "$REPO" rev-parse HEAD)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=$REPO/packages/core:$REPO/packages/atmosphere:$REPO/src
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd "$REPO"
ARGS=""
[ "${WIDEN:-0}" = "1" ] && ARGS="$ARGS --widen-corners"
[ "${DAMPING:-0}" = "1" ] && ARGS="$ARGS --transplant-sentinel-damping"
echo "args: ${ARGS:-<none>}"
$PY scripts/validate/fv3_native/nh_metric_transplant.py $ARGS
rc=$?
echo "  rc=$rc"
# EXIT ON THE PYTHON STATUS. Without -e, a traceback used to be followed
# by this script's own success token and a COMPLETED job state -- which
# is how a run that died in its first control got read as a result.
[ "$rc" -eq 0 ] || { echo "NH_TRANS_FAILED rc=$rc"; exit "$rc"; }
echo NH_TRANS_DONE
