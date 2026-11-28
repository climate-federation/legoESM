#!/bin/bash
#SBATCH --account=glab
#SBATCH --job-name=em_report
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
# Esoteric-climate checkpoint report: ocean heat, zonal state, snapshot RMS, maps.
# usage: DIAG_DIR=<wt>/scripts/validate/esoteric sbatch --output=<log> em_report.sbatch RUN_OUT_DIR DAY [MESH_LEVEL] [RIDGE_LON] [PYTHON]
# Model code comes from the worktree the run recorded (manifest run.cwd); the
# diagnostics come from DIAG_DIR.
set -euo pipefail
OUT=$1; DAY=$2; LEV=${3:-5}; RLON=${4:-0.0}
PY=${5:-/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python}
D=$(printf %04d "$DAY")
HERE=${DIAG_DIR:?set DIAG_DIR to the checkout holding scripts/validate/esoteric (sbatch runs a spooled copy of this file)}
RUN_WT=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['run']['cwd'])" "$OUT/coupled/run_manifest.json")
export PYTHONPATH="$RUN_WT/src:$(ls -d "$RUN_WT"/packages/*/ | sed 's:/$::' | tr '\n' ':')$RUN_WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "model code $RUN_WT @ $(git -C "$RUN_WT" rev-parse HEAD); diagnostics $HERE @ $(git -C "$HERE" rev-parse HEAD)"
CK=$OUT/checkpoint_day_$D.npz
echo "=== OCEAN day $DAY"; $PY "$HERE/ocean_heat.py" "$OUT" "$LEV" "$DAY"
echo "=== ATM day $DAY"; $PY "$HERE/zonal_state.py" "$CK" "$LEV"
$PY "$HERE/eddy_variance.py" "$CK" "$LEV" "$RLON"
CKS=$(ls "$OUT"/checkpoint_day_[0-9]*.npz | awk -v n="$DAY" '{d=$0; sub(/.*checkpoint_day_/,"",d); sub(/\.npz$/,"",d); if (d+0 <= n) print}' | tail -4)
$PY "$HERE/snap_maps.py" "$(dirname "$OUT")/maps_day$D.png" "$LEV" "$RLON" $CKS
